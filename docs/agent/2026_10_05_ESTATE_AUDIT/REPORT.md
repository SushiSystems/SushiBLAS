# Estate audit: sushiblas

**Status:** Complete. Read-only audit of 2026-10-05; nothing in the repository was changed.

One auditing agent per dimension read the tree and a second agent checked each finding against
the files. A finding marked *confirmed* was re-read by the reviewer; *unclear* means the reviewer
could not settle it; *found by review* means the first pass missed it and nobody re-checked it.
Refuted findings are listed and not counted. No build, test or project CLI was run.

| Dimension | High | Medium | Low | Refuted |
| --- | --- | --- | --- | --- |
| Licence | 3 | 10 | 6 | 0 |
| Command line | 2 | 12 | 8 | 0 |
| Documentation | 4 | 11 | 4 | 0 |
| Layout and hygiene | 3 | 12 | 4 | 0 |
| Code shape | 5 | 18 | 2 | 0 |

## Licence

### Facts

- Tree: 361 tracked files at D:/Projects/sushiblas, working tree clean, origin https://github.com/SushiSystems/SushiBLAS.git, no git tags, commits dated 2026-02-08 to 2026-10-04.
- D:/Projects/sushiblas/LICENSE: Apache License 2.0 full text, 190 lines; line 178 reads 'Copyright 2026-present Mustafa Garip & Sushi Systems'.
- D:/Projects/sushiblas/docs/LICENSE: byte-identical copy of the root LICENSE (cmp reports identical).
- D:/Projects/sushiblas/NOTICE.md: attributes GEMM/SYRK tiling ported from portBLAS, 'Copyright (C) Codeplay Software Limited. Licensed under the Apache License, Version 2.0'. No COPYING, no third_party/, no dependencies/ folder tracked; no package-subfolder licence file (cli/ has none).
- Licence history: the repository was MIT ('Copyright (c) 2026 Mustafa Garip & Sushi Systems', commit e8ada31, 2026-02-19) until commit d66c872 (2026-08-01) 'chore(license): relicense from MIT to Apache-2.0'.
- Header variant A (Sushi banner + Apache-2.0 short notice, 22 lines, 'Copyright (c) 2026-present Mustafa Garip & Sushi Systems'): 230 files, identical text in lines 3-22. Examples: include/SushiBLAS/core/common.hpp, src/engine.cpp, tests/common/test_main.cpp.
- Header variant A' (same as A but line 21 is one space short before the closing '*/'): 1 file, kernels/sycl/blas/level1/level1.hpp.
- C/C++ split (305 tracked .cpp/.hpp/.h): 231 carry the Apache header (148 .cpp, 82 .hpp, 1 .h), 74 carry none (72 .cpp, 2 .hpp). All 74 are under tests/: 70 in tests/functional/unit, 2 in tests/functional/integration, 2 in tests/common (reference_reductions.hpp, test_utils.hpp).
- Apache-headed files by area: include/SushiBLAS 44, kernels 38, src 114, examples 2, tests 33. The banner's first line matches the file name in all 231.
- Python: 16 tracked .py files (cli/sushiblas, cli/sushiblas/services, cli/tests), 0 carry a licence header. Examples: cli/sushiblas/__init__.py, cli/sushiblas/__main__.py, cli/sushiblas/cli.py.
- CMake and config: 0 of 10 cmake/*.cmake(.in) files, 0 of 5 CMakeLists.txt, 0 of 4 .toml, Doxyfile, CMakePresets.json and .github/workflows/ci.yml carry a licence header. Examples: cmake/Runtime.cmake, cmake/SushiBLASBuildPolicy.cmake, cmake/SushiBLASCompilerWarnings.cmake.
- No SPDX-License-Identifier line and no @author tag anywhere in the tracked tree.
- Machine-readable declarations: none. cli/pyproject.toml (name sushiblas-cli, version 0.1.0) has no license field and no classifiers; sushi-module.toml has no licence key; CMakeLists.txt:15 project(SushiBLAS VERSION 0.1.0) declares none; Doxyfile declares none; no package.json, Dockerfile, conda or vcpkg.json file is tracked.
- Prose statements: docs/README.md:205-207 ('licensed under the Apache License, Version 2.0'), docs/README.md:202, docs/CONTRIBUTING.md:10-11 (contributions licensed under Apache-2.0), docs/CONTRIBUTING.md:96-121 (header template with Apache notice), docs/ARCHITECTURE.md:35, docs/CHANGELOG.md:461 and :499.
- Copyright holder strings: 'Mustafa Garip & Sushi Systems' with year '2026-present' in LICENSE and all 231 headers; 'Codeplay Software Limited' (no year) in NOTICE.md:10. No other holder appears.
- git shortlog -sne HEAD: 55 Mustafa Garip <mustafa.garip3434@hotmail.com>, 14 Mustafa Garip <51916290+sushimg@users.noreply.github.com>. Committers: the same two identities plus GitHub <noreply@github.com> for the initial commit 8c64d41.
- Co-Authored-By trailers: 46 commits name Claude models (Opus 5 1M 29, Sonnet 5 12, Opus 5 4, Opus 5.5 1) at noreply@anthropic.com; no human co-author.
- cli/pyproject.toml:9 gives the author e-mail mustafagarip@sushisystems.io, which differs from both git identities.
- Third-party code in tree: only the portBLAS-derived algorithm in include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp (rewritten port, Apache-2.0, Codeplay) and the composition in kernels/sycl/blas/level3/syrk.hpp that reuses it; notice kept in NOTICE.md. Nothing is vendored and nothing is fetched (no FetchContent, ExternalProject or GIT_REPOSITORY).
- Build-time dependencies: SushiRuntime (cmake/Runtime.cmake, sibling add_subdirectory or find_package; Sushi Systems' own), SYCL headers from intel-llvm/oneAPI DPC++ (51 includes of <sycl/...>), GoogleTest via find_package (tests/CMakeLists.txt:21, 105 includes), hwloc through SushiRuntime via vcpkg on Windows (cmake/Vcpkg.cmake:3). None has a notice in this repository; cli/sushistack.deps.toml declares no ports.
- Python dependencies (cli/pyproject.toml:11-16): sushicore>=0.7.0, typer>=0.12, rich>=13.0, tomli>=2.0 (py<3.11); build backend setuptools>=61.0; CI adds pytest and pyflakes (.github/workflows/ci.yml:152). No licence record for any of them is kept in the repository.
- CI image: intel/oneapi-basekit:latest (.github/workflows/ci.yml:60); CMakePresets.json:6-22 targets Intel oneAPI DPC++ and its TBB library path.
- docs/CODE_OF_CONDUCT.md:70-76 is adapted from Contributor Covenant 2.1, which is published under CC BY 4.0; the attribution section is present.
- Releases: docs/CHANGELOG.md has only '[Unreleased]' (line 15) and a template heading; Doxyfile:22 says 'v1 (unreleased)'; no tags. The CLI package is installed from the checkout, and no evidence in the tree shows sushiblas-cli was uploaded to PyPI.
- Install rules: no install() of LICENSE or NOTICE.md appears in any CMake file (a search for LICENSE|NOTICE in *.cmake and CMakeLists.txt returns only two comments).

### Corrections from review

- F3 says 69 commits carry one of the two licences. The initial commit 8c64d41 (2026-02-08) contains only README.md; LICENSE first appears in e8ada31 (2026-02-19). Corrected: 68 commits carry a licence (17 under MIT, e8ada31 up to the parent of d66c872; 51 under Apache-2.0), and one commit carries none.
- F12 gives the short line as 73 columns against 74. Measured with CR stripped: kernels/sycl/blas/level1/level1.hpp:21 is 75 characters, the standard line (e.g. src/engine.cpp:21) is 76. The one-space gap is correct.
- F7's title says six prose locations; its own evidence lists seven (docs/README.md:202 and :205-207, docs/CONTRIBUTING.md:10-11 and :96-121, docs/ARCHITECTURE.md:35, docs/CHANGELOG.md:461 and :499).
- The fact 'Third-party code in tree: only the portBLAS-derived algorithm' is incomplete: kernels/sycl/random/philox.hpp:41-49 implements Philox4x32-10 and cites DEShawResearch/random123 include/Random123/philox.h as the reference its constants were checked against (see missed).
- Recomputed and correct: 361 tracked files; 231 Apache-headed C/C++ files (148 .cpp, 82 .hpp, 1 .h) and 74 without (72 .cpp, 2 .hpp); area split 44/38/114/2/33; 16 .py with no header; shortlog 55 + 14; 46 co-authored commits; 51 <sycl/...> includes; LICENSE and docs/LICENSE identical at 190 lines; no tags.

### Findings

#### L1. [high] Apache-2.0 portBLAS port stays under Apache terms after the relicence

Evidence: D:/Projects/sushiblas/NOTICE.md:8-12 ('Copyright (C) Codeplay Software Limited. Licensed under the Apache License, Version 2.0'); D:/Projects/sushiblas/include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp:24-29 ('Tiling/blocking algorithm ported from portBLAS's gemm_local.hpp and gemm_no_local.hpp'); D:/Projects/sushiblas/kernels/sycl/blas/level3/syrk.hpp:24-30 reuses that machinery.

Recommendation: Treat gemm.hpp as a derivative work whose Codeplay-owned portion cannot be relicensed by the owner. Apache-2.0 allows shipping it inside a non-commercial work, but the Apache text, the Codeplay copyright and a 'modified' statement must travel with it. Keep NOTICE.md, add a copy of the Apache-2.0 text for third-party parts (for example LICENSES/Apache-2.0.txt) once the root LICENSE changes, and have the owner decide whether these two files keep an Apache header or get a dual notice. Needs legal confirmation.

Review: confirmed. NOTICE.md:8-12 and gemm.hpp:24-29 read as quoted; syrk.hpp:24-30 says it composes the ported GEMM helpers. Recommendation is sound and flagged for counsel.

#### L2. [high] gemm.hpp header names only the Sushi copyright holder

Evidence: D:/Projects/sushiblas/include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp:9 reads 'Copyright (c) 2026-present Mustafa Garip & Sushi Systems' and nothing else; the Codeplay attribution exists only in a prose comment at lines 24-29 and in NOTICE.md:10.

Recommendation: When headers are rewritten, add the Codeplay Software Limited copyright line and the Apache-2.0 origin to this file's licence block (Apache-2.0 section 4(c) asks that upstream copyright and attribution notices be retained in derivative source). Decide the same for kernels/sycl/blas/level3/syrk.hpp, which NOTICE.md lists under 'Ported files' while calling its logic original.

Review: confirmed. gemm.hpp:9 names only 'Mustafa Garip & Sushi Systems'; Codeplay appears only in the // comment at 24-29 and NOTICE.md:10. git grep for other copyright strings returns NOTICE.md:10 alone.

#### L3. [high] Apache-2.0 and MIT grants already made cannot be withdrawn

Evidence: Root LICENSE has been Apache-2.0 since commit d66c872 (2026-08-01); before that it was MIT (commit e8ada31, 2026-02-19: 'MIT License / Copyright (c) 2026 Mustafa Garip & Sushi Systems'). Origin is https://github.com/SushiSystems/SushiBLAS.git. 69 commits carry one of the two licences.

Recommendation: Record in the relicensing design document that every commit up to the switch stays available under MIT (before 2026-08-01) or Apache-2.0 (after), so anyone holding a copy may keep using and commercialising that snapshot. The new licence binds only later commits. No tag or release exists, so mark the cut-over commit with a tag. Whether the GitHub repository is public was not verifiable from the checkout.

Review: confirmed. e8ada31 LICENSE is MIT, d66c872 (2026-08-01) switched to Apache-2.0, no tags. One number is off: the initial commit 8c64d41 holds only README.md and no LICENSE, so 68 commits carry a licence (17 MIT, 51 Apache), not 69. The conclusion stands.

#### L4. [medium] 74 C++ test files carry no licence header

Evidence: 72 .cpp and 2 .hpp without any copyright or licence text: 70 in D:/Projects/sushiblas/tests/functional/unit (e.g. test_abs.cpp), D:/Projects/sushiblas/tests/functional/integration/test_dependency_stress.cpp, test_framework_sim.cpp, D:/Projects/sushiblas/tests/common/reference_reductions.hpp, test_utils.hpp. 33 other files under tests/ do carry the header (e.g. tests/common/test_main.cpp). docs/CONTRIBUTING.md:96-98 says 'Every file starts with this signature'.

Recommendation: Give all 74 files the new header in the same pass that rewrites the other 231, so the tests directory has one shape.

Review: confirmed. Recounted: 72 .cpp + 2 .hpp without header, 70 in tests/functional/unit, 2 in tests/functional/integration, 2 in tests/common; 33 headed files under tests/. CONTRIBUTING.md:98 says 'Every file starts with this signature'.

#### L5. [medium] No Python, CMake or config file carries a licence header

Evidence: 0 of 16 .py files (e.g. D:/Projects/sushiblas/cli/sushiblas/cli.py, cli/sushiblas/__init__.py, cli/tests/), 0 of 10 files in D:/Projects/sushiblas/cmake/, 0 of 5 CMakeLists.txt, plus Doxyfile, sushi-module.toml, cli/config.toml, cli/sushistack.deps.toml, .github/workflows/ci.yml. The source-comments skill requires a licence block before the module docstring in Python.

Recommendation: Add the new licence block to the 16 Python files. Have the owner decide whether CMake and TOML files get a short header or an SPDX line; apply the same answer in every Sushi repository.

Review: confirmed. Recounted 0/16 .py, 0/9 .cmake + 1 .in, 0/5 CMakeLists.txt, 0/4 .toml, Doxyfile, ci.yml. source-comments skill: the Python module docstring opens the file 'after the license block'. The skill names no rule for CMake/TOML, so leaving that to the owner is right.

#### L6. [medium] No machine-readable licence declaration exists anywhere

Evidence: D:/Projects/sushiblas/cli/pyproject.toml:5-16 has no 'license' key and no classifiers; D:/Projects/sushiblas/sushi-module.toml:8-12 has no licence key; no SPDX-License-Identifier line in any tracked file (git grep for licen[sc]e|classifiers|SPDX over *.toml, *.cmake, CMakeLists.txt, Doxyfile, *.json, *.in, *.yml, *.py returns nothing).

Recommendation: Declare the new licence in cli/pyproject.toml (a 'license' expression or LicenseRef plus license-files, and a 'License :: Other/Proprietary License' classifier since the new licence is not OSI-approved) and, if the module manifest format gains a licence field, in sushi-module.toml. Nothing here disagrees today; the gap is that pip metadata would report no licence.

Review: confirmed. cli/pyproject.toml (27 lines) has no license key or classifiers; sushi-module.toml:8-12 has name, alias, distribution, repo only; no SPDX line outside the docs. The cited line ranges match.

#### L7. [medium] Six prose locations state Apache-2.0 and will go stale

Evidence: D:/Projects/sushiblas/docs/README.md:205-207; D:/Projects/sushiblas/docs/CONTRIBUTING.md:10-11 (inbound contributions 'licensed under the project's Apache License, Version 2.0') and :96-121 (header template); D:/Projects/sushiblas/docs/README.md:202; docs/ARCHITECTURE.md:35 and docs/CHANGELOG.md:461, :499 (these three describe portBLAS and stay true).

Recommendation: Rewrite docs/README.md:205-207 and the CONTRIBUTING header template for the new licence. Replace CONTRIBUTING.md:10-11 with an inbound term that lets the owner keep relicensing rights (a CLA or copyright assignment), because 'inbound = outbound' under a non-commercial licence would stop the owner from selling commercial licences for contributed code.

Review: confirmed. All cited lines exist and read as described. The title says six locations but the evidence lists seven (README 202, 205-207; CONTRIBUTING 10-11, 96-121; ARCHITECTURE 35; CHANGELOG 461, 499). The CLA advice goes past measurement into licensing strategy, yet it follows from the owner's stated aim.

#### L8. [medium] Duplicate licence text in docs/LICENSE

Evidence: D:/Projects/sushiblas/docs/LICENSE is byte-identical to D:/Projects/sushiblas/LICENSE (190 lines each, cmp identical).

Recommendation: Keep one licence file at the root and delete docs/LICENSE, or the two will drift at the relicence. Deleting it needs the owner's approval.

Review: confirmed. cmp LICENSE docs/LICENSE reports identical; 190 lines; both tracked.

#### L9. [medium] No root README, so the repository front page states no licence

Evidence: D:/Projects/sushiblas has no README.md at the root (git ls-files top level: .github, .gitignore, cli, cmake, CMakeLists.txt, CMakePresets.json, docs, Doxyfile, examples, include, kernels, LICENSE, NOTICE.md, src, sushi-module.toml, tests). Commit d66c872 moved README.md to docs/README.md.

Recommendation: Add a root README.md as the front door, with the licence paragraph stating who may use the code and that commercial use needs a separate agreement.

Review: confirmed. No root README.md in git ls-files; d66c872 --stat shows 'README.md => docs/README.md'. Also a gap against the documentation architecture in the global CLAUDE.md (README.md is the front door).

#### L10. [medium] Copyright holder is a joint name with no defined legal entity

Evidence: 'Mustafa Garip & Sushi Systems' in D:/Projects/sushiblas/LICENSE:178 and in all 231 headers (e.g. include/SushiBLAS/core/common.hpp:9). CONTRIBUTING.md:3 calls the project 'a personal project developed by Mustafa Garip'. The year form is '2026-present'.

Recommendation: Before rewriting 231 headers, have the owner fix the holder string once for all repositories: either the person, or the registered company name. A non-commercial licence is enforced by the holder, so the holder must be a legal person that exists. The same string then goes into LICENSE, headers and pyproject.

Review: confirmed. LICENSE:178 and header line 9 carry 'Mustafa Garip & Sushi Systems', year '2026-present'; CONTRIBUTING.md:3 says 'a personal project developed by Mustafa Garip'. Whether Sushi Systems is a registered entity cannot be read from the tree; the audit says so by asking the owner.

#### L11. [medium] Philox kernel cites Random123 and has no third-party record

Evidence: D:/Projects/sushiblas/kernels/sycl/random/philox.hpp:41-49 ('Round constants verified against the reference Random123 implementation (DEShawResearch/random123, include/Random123/philox.h)'), constants at :65-68; also docs/ARCHITECTURE.md:726 and docs/CHANGELOG.md:483. NOTICE.md does not mention Random123 or D. E. Shaw Research, and the audit searched only for portBLAS and Codeplay.

Recommendation: Have the owner state whether philox.hpp was written from the SC'11 paper or adapted from Random123 source (BSD-3-Clause, to be read from upstream). If adapted, add the D. E. Shaw Research copyright and licence text to NOTICE.md before the relicence; if written from the paper, record that in NOTICE.md so the provenance question is closed. Run the same search for every kernel that names an external reference, not only portBLAS.

Review: found by review.

#### L12. [medium] The tree says sushicore is taken from PyPI, which the audit does not record

Evidence: D:/Projects/sushiblas/docs/CHANGELOG.md:21 ('Took `sushicore` from PyPI instead of a sibling checkout'), docs/agent/reports/2026-09-22-wave-1-task-7d.md:1 and :29, cli/pyproject.toml:12 ('sushicore>=0.7.0'). .github/workflows/ci.yml:141 and docs/CLI_GUIDE.md:28 still say sushicore 'is published to no index'.

Recommendation: Record that this repository's own Python dependency is a first-party package already published on PyPI, so its Apache-2.0 releases stay installable after the relicence and the open lower bound '>=0.7.0' will pull whichever licence the newest release carries. Decide the pin when sushicore relicenses, and correct the two stale 'published to no index' statements in the same change. The PyPI index itself was not queried.

Review: found by review.

#### L13. [medium] The MIT to Apache-2.0 relicence was never entered in the changelog

Evidence: git log shows d66c872 (2026-08-01) 'chore(license): relicense from MIT to Apache-2.0'; a search of D:/Projects/sushiblas/docs/CHANGELOG.md for licen[sc]e returns only lines 461 and 499, both about the portBLAS notice.

Recommendation: When the new relicence lands, add one changelog line for it and one for the earlier MIT to Apache-2.0 switch, each naming the cut-over commit, so the licence that applies to any snapshot can be read from the repository without git archaeology. This is the record F3 asks for, placed where the documentation rules put it.

Review: found by review.

#### L14. [low] Commits co-authored by AI models

Evidence: 46 of 69 commits carry 'Co-Authored-By: Claude ... <noreply@anthropic.com>' trailers (29 Opus 5 1M, 12 Sonnet 5, 4 Opus 5, 1 Opus 5.5). git shortlog lists only Mustafa Garip under two e-mail addresses; no other human contributor.

Recommendation: No third human holds copyright, so the owner can relicence without anyone's consent. Note for counsel that parts of the code are machine-generated, since copyright protection of such output varies by jurisdiction and a licence can only restrict what copyright covers.

Review: confirmed. Recounted trailers: 29 + 12 + 4 + 1 = 46 distinct commits of 69; shortlog shows only the two Mustafa Garip identities.

#### L15. [low] One header deviates from the common block by one space

Evidence: D:/Projects/sushiblas/kernels/sycl/blas/level1/level1.hpp:21 reads '/* permissions and limitations under the License.                        */' (73 columns, the other 230 files have 74).

Recommendation: The header rewrite replaces this line anyway; a replacement script that matches the exact Apache block must not skip this file.

Review: confirmed. level1.hpp:21 is one space short of the other headers. The column figures are off: the line is 75 characters and the standard one 76 (CR stripped), not 73 and 74. The one-space difference and the warning for a replace script hold.

#### L16. [low] Toolchain dependencies have their own terms and no record here

Evidence: D:/Projects/sushiblas/.github/workflows/ci.yml:60 uses intel/oneapi-basekit:latest (Intel EULA, and an unpinned tag); D:/Projects/sushiblas/CMakePresets.json:6-22 builds against Intel oneAPI DPC++ and TBB; tests/CMakeLists.txt:21 uses GoogleTest (BSD-3-Clause); hwloc (BSD-3-Clause) arrives through SushiRuntime; typer (MIT), rich (MIT), tomli (MIT) in cli/pyproject.toml:13-15. Licences named from general knowledge, not read from source in this audit.

Recommendation: None of these is copied into the repository, and none is GPL or LGPL, so a non-commercial outbound licence does not conflict with them. If binaries are ever distributed, the SYCL runtime, TBB, hwloc and GoogleTest notices must ship with them; add a third-party list (docs/reference or NOTICE.md) with each licence read from its source. Pin the CI image tag.

Review: confirmed. ci.yml:60 intel/oneapi-basekit:latest, CMakePresets.json:6-22 icx-cl and tbb paths, tests/CMakeLists.txt:21 GTest, pyproject.toml:13-15 all as cited. Licences are from general knowledge, which the audit states. It omits that the dependencies skill rejects vendor SDK licences that restrict redistribution, a row the Intel oneAPI EULA should be read against.

#### L17. [low] The dependencies skill rejects the licence class the owner is moving to

Evidence: C:/Users/sushi/.claude/skills/dependencies/SKILL.md licence table: 'Non-commercial, source-available, field-of-use limits | Rejected'. D:/Projects/sushiblas/cli/pyproject.toml:12 depends on sushicore and cli/sushistack.deps.toml:25 on sushiruntime, both Sushi Systems repositories due for the same relicence.

Recommendation: After the relicence, the rule as written would reject Sushi's own modules as dependencies of each other. Amend the skill to exempt first-party Sushi Systems packages, in the same change that introduces the new licence.

Review: confirmed. dependencies skill table has the row 'Non-commercial, source-available, field-of-use limits | Rejected'; pyproject.toml:12 sushicore, sushistack.deps.toml:25 depends_on sushiruntime. The fix (one first-party exemption in the skill) is one change in one place.

#### L18. [low] Author e-mail in package metadata differs from the commit identities

Evidence: D:/Projects/sushiblas/cli/pyproject.toml:9 'mustafagarip@sushisystems.io'; git shortlog: mustafa.garip3434@hotmail.com (55) and 51916290+sushimg@users.noreply.github.com (14).

Recommendation: Add a .mailmap that maps both commit addresses to the canonical identity, so authorship evidence for the relicence reads as one person.

Review: confirmed. pyproject.toml:9 mustafagarip@sushisystems.io; shortlog 55 hotmail + 14 GitHub noreply; no .mailmap tracked.

#### L19. [low] Installed package ships neither LICENSE nor NOTICE

Evidence: No install() rule mentions LICENSE or NOTICE in D:/Projects/sushiblas/src/CMakeLists.txt or any cmake/*.cmake (search for LICENSE|NOTICE returns only comments at src/CMakeLists.txt:210 and cmake/SushiBLASLogging.cmake:37), although src/CMakeLists.txt:215 makes the library consumable through find_package(SushiBLAS).

Recommendation: Install LICENSE and NOTICE.md beside the headers, since the installed include tree contains the portBLAS-derived gemm.hpp. This is a build-configuration change and needs the owner's approval.

Review: confirmed. install() rules: CMakeLists.txt:107 (config files), src/CMakeLists.txt:220, 228, 232; none installs LICENSE or NOTICE.md, and :228 installs include/ which holds gemm.hpp. The audit did not list the root CMakeLists.txt:107 rule, but it installs no licence file either.

### Not checked

- Whether the GitHub repository SushiSystems/SushiBLAS is public, and whether any GitHub release, fork or mirror exists (no network access was used).
- Whether sushiblas-cli 0.1.0 was ever uploaded to PyPI; the tree shows no publish step, but the index itself was not queried.
- The actual licence texts of typer, rich, tomli, setuptools, pytest, pyflakes, GoogleTest, hwloc, intel-llvm/oneAPI and TBB; F13 names them from general knowledge.
- The licence and notice state of sushiruntime and sushicore, which this repository depends on; they belong to their own audits.
- A line-by-line comparison of gemm.hpp against upstream portBLAS to measure how much Codeplay expression survives the rewrite.
- Whether syrk.hpp or any file other than gemm.hpp contains code copied from portBLAS or from another project without attribution; only explicit mentions of portBLAS and Codeplay were searched.
- The untracked build/ directory and anything else ignored by git.
- Headers of files in git history that no longer exist at HEAD.
- Doxyfile content beyond the PROJECT_* and INPUT keys, and the full text of docs/SECURITY.md, docs/INTEGRATION.md, docs/CLI_GUIDE.md and docs/GPU_VALIDATION.md beyond a search for licence words.
- Legal validity of the proposed non-commercial licence and of relicensing AI-co-authored code; these need counsel.

## Command line

### Facts

- Package `sushiblas-cli` 0.1.0 (D:/Projects/sushiblas/cli/pyproject.toml); console scripts `sb` and `sushiblas`, both `sushiblas.cli:app`; `python -m sushiblas` also works (cli/sushiblas/__main__.py).
- Framework: Typer >=0.12 with rich >=13; `no_args_is_help=True`, `add_completion=False`, `rich_markup_mode="rich"`; root group class is `sushicore.typer_help.help_group(console.current)`.
- Requires `sushicore>=0.7.0`; sushicore's own pyproject is at 0.7.0. `requires-python = ">=3.10"` with a `tomli` fallback.
- Command tree, local (cli/sushiblas/cli.py): `build` [--type/-t release|debug|relwithdebinfo, --clean, --define/-D VAR=VALUE repeatable]; `test` [--suite/-s unit|regression|integration|functional|all|package (default functional), --filter/-f, --repeat/-r N]; `run` [TARGET] [--sort] plus args after `--`; `clean`; `doxygen`; `config`; `env` [--all].
- Command tree, from sushicore (`register_provision_commands`, sushicore/provision/commands.py): `setup` [--dry-run, --yes, --toolchain repeatable, --no-gpu]; `doctor` [--for GROUP, groups build|test|infer|eval]; `link` [--workspace]; `unlink` [--workspace].
- No root callback: `sb` has no `--version`, no `--json`, no `--describe`. `se` has `--version` (sushiengine/cli/sushiengine/cli.py:114); `hub` has `--json` and `--describe` (sushistack/cli/sushihub/cli.py:54-57).
- Help panels: Project (build, test, run, clean), Documentation (doxygen), Diagnostics (config, env, setup, doctor, link, unlink). Every command has a docstring and an `epilog` of examples; every option has help text.
- Taken from sushicore: LazyConsole, help page, ModuleProfile, ModuleConfig, StackConfig, StackBuildEnv, Diagnostics (`config`/`env` bodies), ExecutableIndex, Runner, CMakeDriver, cmake_cache, toolchain_args, the four provision commands.
- Local to sb: the Typer declarations of build/test/run/clean/doxygen/config/env, the `BuildType` and `Suite` enums, the configure-argument assembly, the SushiRuntime sibling check and the whole `--suite package` consumer flow (cli/sushiblas/services/project.py).
- Layout: `cli/sushiblas/cli.py` holds every command and `services/` does the work; there is no `commands/` package. Services import `console` and print directly.
- Rendering: Rich through the sushicore console (header, info, success, error, command echo). The JSON renderer exists in sushicore but sb never sets `LazyConsole.machine`, so there is no machine-readable output.
- Colour: `[cli] color = auto|always|never` in cli/config.toml or config.local.toml, `SUSHI_CLI_COLOR`, and `NO_COLOR` which always wins (sushicore/config.py:112). No `--no-color` flag. The logo shows only on a UTF-8 terminal with 256 colours or more.
- Exit codes: child process code passed through for cmake, ctest and the run target; 1 for missing build/, missing SushiRuntime sibling, unresolved executable, missing Doxyfile or doxygen; 2 for a malformed -D, for --filter/--repeat with --suite package, and for Typer usage errors; `clean` always 0.
- Errors: handled failures print through `console.error` and return a code. Outside a checkout `find_project_root` raises `SystemExit(message)`, which prints a plain unstyled line and exits 1 (sushicore/module_config.py:75). No top-level exception guard; Typer's default pretty traceback is left on.
- Ctrl+C: `Runner(console, PROFILE.program)` is built with the default `catch_interrupt=False` (project.py:78, sushicore/proc.py:48), so an interrupt during a build or test propagates rather than returning 130.
- Help wording: imperative, one sentence, capitalised, ends with a period ("Configure and build the project.", "Remove the build/ tree."). The sushicore commands use the same register.
- Config precedence: built-in defaults, cli/config.toml, workspace-shared config, cli/config.local.toml, then `SB_*` environment variables plus unprefixed `SUSHIRUNTIME_DIR` (cli/sushiblas/config.py:29-41).
- The repository has no root README.md; the front-door text is D:/Projects/sushiblas/docs/README.md. There is no docs/guides/ directory; the CLI manual is docs/CLI_GUIDE.md.

### Corrections from review

- Fact 'Every command has a docstring and an epilog of examples' is wrong for one command: `unlink` is registered with `@command("unlink")` and no examples (sushicore/sushicore/provision/commands.py, the unlink declaration after line 315), so its epilog is empty. Corrected: 10 of 11 commands carry examples.
- F2 'breaks every command' is too wide: `sb --help` and `sb <cmd> --help` survive a malformed config or unknown theme, because sushicore/typer_help.py:82-93 catches Exception around the console build and falls back to Typer's own help with a logged warning. Corrected: every command body that touches the console or load_config fails; help does not.
- F1 'match ... line for line (same flags, same help strings)': flags match, but the --suite help text and the test docstring differ between sb (cli.py:80-82, 90-99) and sa (sushiai cli.py:79-81, 89-96). The conclusion is unchanged.
- Exit-code fact is incomplete: exit 2 is also returned by `sb setup` with no SushiRuntime checkout or an unknown --toolchain (commands.py:193-203), by `sb doctor --for <unknown>` and by `sb link`/`unlink` with no workspace; `link`/`unlink` return 1 on LinkEditError.
- not_checked item 'CLI_GUIDE.md:257 --toolchain oneapi is unverified' is resolved: `oneapi` is a valid key (sushicore/provision/selection.py:26-32; keys are intel-llvm, adaptivecpp, oneapi).
- Recomputed and correct: console scripts sb and sushiblas (cli/pyproject.toml:18-20); sushicore requirement 0.7.0 against sushicore/pyproject.toml:7 version 0.7.0; command tree of 7 local plus 4 sushicore commands with the listed options; no root README.md; panel assignment Project(4)/Documentation(1)/Diagnostics(6).

### Findings

#### C1. [high] build/test/run/clean/doxygen/config/env are declared per CLI instead of coming from sushicore

Evidence: D:/Projects/sushiblas/cli/sushiblas/cli.py:40-166 declares seven commands locally. The option declarations match D:/Projects/sushiai/cli/sushiai/cli.py:48-56, 76-86, 109-114, 242-244 line for line (same flags, same help strings). `BuildType`, `Suite` and `_CMAKE_BUILD_TYPE` are defined again in cli/sushiblas/services/project.py:34-71. Only setup/doctor/link/unlink are registered from sushicore (cli.py:169-177).

Recommendation: Owner decision on the interface: add a sushicore registrar for the common build commands, shaped like `register_provision_commands`, taking the module's build policy as an object. Until then every wording or flag change has to be made in each CLI by hand.

Review: confirmed. sushiblas/cli/sushiblas/cli.py:40-166 declares seven commands; sushiai/cli/sushiai/cli.py:48-59, 109-115 carry the same options, down to the SB_SYCL_TARGETS example pasted into sa's help. 'Line for line' overstates it: the --suite help and the test docstring differ (sb cli.py:80-82 vs sa 79-81). SOLID: a registrar shaped like register_provision_commands that takes a build-policy object is a brick-shaped fix, and the audit correctly routes the interface to the owner.

#### C2. [high] A malformed config file or unknown theme reaches the user as a traceback

Evidence: sushicore/workspace.py:42-47 and sushicore/config.py:55-59 call `tomllib.load` with no handler; sushicore/theme.py:109 raises `ValueError` for an unknown `[cli] theme`, called from `build_console` (sushicore/__init__.py:68). sb has no guard: cli/sushiblas/cli.py:66,100,119 call the service directly and `app` is the console script. A syntax error in cli/config.local.toml, a file `sb setup` tells users is "Safe to edit", breaks every command.

Recommendation: Catch `tomllib.TOMLDecodeError` and the theme `ValueError` at the sushicore boundary and re-raise one config error naming the file; give every CLI the same entry point that prints one line and exits non-zero.

Review: confirmed. sushicore/workspace.py read_toml and config.py:55-59 call tomllib.load bare; theme.py get_theme raises ValueError; sb cli.py has no guard. One correction: help pages survive, because typer_help._write_guarded (typer_help.py:82-93) catches Exception and falls back to Typer's help, so 'breaks every command' should read 'every command that does work'. Recommendation is brick-shaped (one config error at the sushicore boundary, one shared entry point).

#### C3. [medium] No --version, --json or --describe on sb

Evidence: cli/sushiblas/cli.py:27-34 defines no callback. `se --version` exists at D:/Projects/sushiengine/cli/sushiengine/cli.py:103-115; `hub --json` and `hub --describe` at D:/Projects/sushistack/cli/sushihub/cli.py:54-58. sushicore already carries the JSON renderer (`LazyConsole.machine`, sushicore/cli_console.py:39-49) but sb never sets it.

Recommendation: Decide the root flags once for all seven CLIs and provide them from sushicore, so `sb --version` reports the CLI version, project root and sushicore version the way `se` does.

Review: confirmed. sb cli.py:27-34 has no callback; se cli.py:103-115 has --version; hub cli.py:49-57 has --json and --describe; LazyConsole.machine exists at sushicore/cli_console.py:39-49 and sb never sets it.

#### C4. [medium] `sb run --sort` keeps a flag name sushiengine has retired

Evidence: cli/sushiblas/cli.py:115-116 `--sort` with help "Interactively pick an executable." sushiengine renamed it to `--pick` and keeps `--sort` hidden as a dated alias (D:/Projects/sushiengine/cli/sushiengine/cli.py:252-254, 957). CLI_GUIDE.md:192 and the epilog at cli.py:107 document `--sort`.

Recommendation: Rename to `--pick` with a hidden `--sort` alias, matching se, and update CLI_GUIDE.md. This is a public surface change, so it needs the owner's sign-off and a changelog entry.

Review: confirmed. sb cli.py:115-116 and epilog :107, CLI_GUIDE.md:192 use --sort; se cli.py:250-254 has --pick with a hidden --sort. sa (sushiai cli.py:113-114) still uses --sort too, so the rename belongs in the shared run command of F1 rather than in sb alone.

#### C5. [medium] `doxygen` command name and panel names differ from sibling CLIs

Evidence: sb: `doxygen` in a panel of its own, "Documentation" (cli.py:22, 132-134), and config/env/setup/doctor under "Diagnostics" (cli.py:23). se: the command is `docs`, with `doxygen` as a dated alias (sushiengine cli.py:294, 959), and config/env sit under "Environment" (cli.py:45, 834). sr and sa put `doxygen` under "Project" (sushiruntime cli.py:206-207, sushiai cli.py:131-132); sd puts it under "Build" and uses "Environment" (sushidsp cli.py:23-25, 127-128).

Recommendation: Fix one panel vocabulary and one name for the API-reference command across the CLIs, then apply it here. "Documentation" as a one-command panel exists only in sb.

Review: confirmed. Panels checked: sb Project/Documentation/Diagnostics (cli.py:21-23); se Build/.../Environment with command 'docs'; sr and sa put doxygen under Project; sd under Build with Environment. st uses Project/Benchmark/Diagnostics (sushitrack_cli/cli.py:36-38), which the audit did not list.

#### C6. [medium] Short options and defaults differ between sb and sd for the same commands

Evidence: `env --all` has no short form in sb (cli.py:162-163) and `-a` in sd (sushidsp cli.py:120). `-r` means `--repeat` in sb test (cli.py:86) and `--report` in sd check (sushidsp cli.py:144). `test --suite` defaults to `functional` in sb (cli.py:79) and `unit` in sd (sushidsp cli.py:94). sd `test` and `run` take `--type`; sb's do not. sb has no `check`, `status` or `where`, which sd or se have (sushidsp cli.py:137, sushiengine cli.py:862, 874).

Recommendation: Write the shared option table (name, short form, default) in sushicore's documentation and bring sb and sd to it. Whether sb needs `check` is the owner's call.

Review: confirmed. sd cli.py:120 '-a', :144 '-r' for --report, :94 Suite.unit default, --type on test and run; sb cli.py:79, 86, 162-163 as stated. se status/where exist at the cited area.

#### C7. [medium] Ctrl+C during build or test is not turned into the 130 exit sushicore offers

Evidence: cli/sushiblas/services/project.py:78 `_RUNNER = Runner(console, PROFILE.program)` leaves `catch_interrupt` at its default False (sushicore/proc.py:48), so the `KeyboardInterrupt` branches at proc.py:100-103 and 135-140 re-raise. The ctest child is not terminated by the CLI in that path.

Recommendation: Pass `catch_interrupt=True`, or make it the default in sushicore, so an interrupted `sb build` prints "Interrupted." and exits 130.

Review: confirmed. project.py:78 builds Runner with the default; proc.py:48 catch_interrupt=False; proc.py:100-103 and 135-140 re-raise, and proc.terminate() is skipped. sr, sa, sd and se all construct Runner the same way; only st passes catch_interrupt=True (sushitrack_cli/process.py:25-26), so changing the sushicore default is the single-point fix, not a per-CLI edit.

#### C8. [medium] CLI_GUIDE.md links to a root README.md that does not exist

Evidence: D:/Projects/sushiblas/docs/CLI_GUIDE.md:10 `[README.md](../README.md)` and :283 `[README.md](../README.md#building-with-cmake-directly)`. A glob for README.md at the repository root returns nothing; the text lives in docs/README.md (the "Building with CMake directly" heading is at docs/README.md:85).

Recommendation: Add the root README.md the documentation architecture requires, or point both links at `README.md` in the same folder until the layout is rebuilt.

Review: confirmed. ls of D:/Projects/sushiblas shows no README.md at the root; CLI_GUIDE.md:10 and :283 link ../README.md; the heading is at docs/README.md:85. The second option offered (repoint the links 'until the layout is rebuilt') is an interim patch; the root README is the fix the documentation architecture requires.

#### C9. [medium] CLI_GUIDE offers `pipx install ./cli`, which its own paragraph says cannot resolve sushicore

Evidence: docs/CLI_GUIDE.md:27-29 states sushicore "is published to no index, so pipx cannot resolve it as an ordinary dependency"; lines 34-36 then give `pipx install ./cli  # or: pip install ./cli`. cli/pyproject.toml:12 requires `sushicore>=0.7.0`.

Recommendation: Remove the direct install lines or state the extra step (install sushicore from its checkout first). Keep `hub install-cli sushiblas` as the single documented path.

Review: confirmed. CLI_GUIDE.md:27-36 says pipx cannot resolve sushicore and then offers 'pipx install ./cli'; pyproject.toml requires sushicore>=0.7.0.

#### C10. [medium] Config comments and README describe a toolchain lookup the code no longer performs

Evidence: cli/config.toml:12-14 says cxx and vcpkg_root resolve "from the runtime sibling's `dependencies/` tree", and lines 37-39 give `sushiruntime/dependencies/...` examples; cli/sushiblas/services/project.py:6-8 and docs/README.md:14, 50 say the same. The code searches the workspace `dependencies/` and then the machine root (sushicore/stack_config.py:82-99, 105-139); the runtime checkout is not consulted. cli/config.toml:4 also lists "CLI flags" as a precedence layer; no sb flag overrides a config field.

Recommendation: Rewrite the config.toml header, the project.py docstring and the two README passages to name the workspace tree and `~/.sushisystems`, as CLI_GUIDE.md:268-271 already does, and drop the "CLI flags" layer.

Review: confirmed. cli/config.toml:4, 12-14, 37-39, project.py:6-8 and docs/README.md:14, 50 as quoted; sushicore/stack_config.py:82-99 builds the roots from SUSHISTACK_DEPS_DIR, the workspace dependencies/ and provision_home.search_roots(), never from the runtime checkout.

#### C11. [medium] `sb build --type` deletes the whole build tree without --clean and without saying so

Evidence: D:/Projects/sushiblas/cli/sushiblas/services/project.py:188-193: when needs_configure returns true (sushicore/cmake_driver.py:51-74, which fires when the cached CMAKE_BUILD_TYPE differs from the requested one) the service runs shutil.rmtree(build_dir, ignore_errors=True) after printing only 'Configuring CMake... (type=...)'. That removes build/package/ and every -D cache variable too, although the build docstring (cli.py:62-64) and CLI_GUIDE.md:88, 104-105 say variables persist and only --clean starts over. sa and sr carry the same line (sushiai project.py:215, sushiruntime project.py:500).

Recommendation: Print that the tree is being removed and why (through CMakeDriver, so every CLI says it the same way), and state in the --type help and CLI_GUIDE.md that switching the build type starts from an empty tree. Whether to keep one tree per build type instead is the owner's call.

Review: found by review.

#### C12. [medium] The service layer prints to the terminal and the package has no commands/ split; the audit recorded this as a fact, not a finding

Evidence: python-code-style: 'commands/ parses arguments, services/ does the work. A service never imports the command framework or writes to the terminal', 'no module-level singletons', enum members UPPER_SNAKE. D:/Projects/sushiblas/cli/sushiblas/services/project.py:28 imports console and prints throughout (171, 183, 231, 240...); :78-79 are module-level singletons _RUNNER and _DRIVER; :34-52 enum members are lowercase (release, unit); all commands sit in cli/sushiblas/cli.py with no commands/ package; discovery.py:19 and diag.py:34 are further module singletons.

Recommendation: Settle this in the shared build-command registrar of F1 rather than per CLI: services return a result or raise the module's error, the sushicore-provided command renders it. Record it as a finding so the F1 design carries it.

Review: found by review.

#### C13. [medium] CONTRIBUTING.md names `sb test --suite all` as the full suite that gates a change, which skips the package suite

Evidence: D:/Projects/sushiblas/docs/CONTRIBUTING.md:74 and :80 ('The full test suite passes (`sb test --suite all`)'); project.py:244-249 excludes the package consumer from `all`. The audit listed CONTRIBUTING.md as only grepped and raised the same mismatch for docs/README.md alone (F12).

Recommendation: Reword the gate to 'every CTest test' and add `sb test --suite package` for changes that touch public headers or install rules, matching CLI_GUIDE.md:181-182.

Review: found by review.

#### C14. [medium] Colour, theme and exit-code behaviour are undocumented in the CLI manual

Evidence: D:/Projects/sushiblas/docs/CLI_GUIDE.md:75-78 mentions only `[cli] background`; nothing in CLI_GUIDE.md or docs/README.md names NO_COLOR, SUSHI_CLI_COLOR/THEME/ICONS (sushicore/config.py:103-113) or the exit codes the audit lists as facts (1, 2, child pass-through). cli/config.toml:42-46 points to 'sushicore's README' instead.

Recommendation: Add one shared section, written once in sushicore's documentation and linked from each CLI guide, covering colour controls and the exit-code table.

Review: found by review.

#### C15. [low] Dependency fragment header describes only hub, not `sb setup`

Evidence: cli/sushistack.deps.toml:2-5, 9, 18 say `hub` aggregates the fragment and that descriptions are "shown by `hub doctor`". `sb setup` and `sb doctor` read the same fragment (cli.py:169-177, sushicore/provision/commands.py:109-123).

Recommendation: Reword the header so it names the module's own `setup`/`doctor` as well as hub.

Review: confirmed. cli/sushistack.deps.toml:2-5, 9, 18 name only hub; cli.py:169-177 registers setup/doctor over the same fragment.

#### C16. [low] docs/README.md describes `--suite all` as everything and omits the package suite

Evidence: docs/README.md:117 `sb test --suite all  # everything`; the list at 113-119 has no `--suite package`. The code excludes package from `all` and says so at run time (services/project.py:244-249); CLI_GUIDE.md:173-176 documents it correctly.

Recommendation: Change the comment to "every CTest test" and add the package line.

Review: confirmed. docs/README.md:117 '# everything', no package line in 113-119; project.py:244-249 prints the exclusion. docs/CONTRIBUTING.md:74 and :80 repeat the same claim (see missed).

#### C17. [low] Unused re-exports in the discovery service

Evidence: cli/sushiblas/services/discovery.py:21 `find_executables` and :26-28 `select_interactive` have no caller in the repository (grep over *.py finds only the definitions). `run` goes through `discovery.resolve` (project.py:384) and the package suite through `match_by_name` (project.py:356).

Recommendation: Delete both names, with the owner's approval since the code predates this audit.

Review: confirmed. grep over *.py in the repository, tests included, finds find_executables and select_interactive only at services/discovery.py:21 and :26.

#### C18. [low] The default run target is stated in three places

Evidence: `default_target="sushiblas_example"` in cli/sushiblas/config.py:35, `target_bin: str = "sushiblas_example"` in config.py:70, and `target_bin` in cli/config.toml:18 and :31. config.py:26-28 says the profile exists so these "cannot disagree".

Recommendation: Derive the dataclass default from `PROFILE.default_target` and keep the config.toml entry only as an override.

Review: confirmed. config.py:35 and :70, cli/config.toml:18 and :31. The Windows entry is a fourth spelling ('sushiblas_example.exe'). Deriving the dataclass default from PROFILE.default_target is a sound single-source fix.

#### C19. [low] The package suite can raise OSError out of the DLL copy

Evidence: cli/sushiblas/services/project.py:278-279 `shutil.copy2(dll, dest / dll.name)` has no handler; a locked or read-only DLL beside the consumer ends `sb test --suite package` with a traceback. The path is Windows-only (project.py:271-272).

Recommendation: Catch `OSError`, report the file through `console.error` and return 1.

Review: confirmed. project.py:278-279 copy2 without a handler; Windows-only by :271-272. Not executed.

#### C20. [low] `sb run --sort` without a terminal ends in an unhandled EOFError

Evidence: sushicore/discovery.py:134-137 calls `IntPrompt.ask` with no EOF handling, reached from cli/sushiblas/services/project.py:384. Other sushicore prompts handle it (sushicore/renderer.py:163, 234). Inferred from reading; not executed.

Recommendation: Handle `EOFError` in `ExecutableIndex.select` and return None with a one-line message.

Review: confirmed. sushicore/discovery.py:134-137 IntPrompt.ask with no EOFError handler; renderer.py:163 and :234 handle it. Inferred from source, as the audit says. The fix sits in sushicore's ExecutableIndex.select, which is the right brick.

#### C21. [low] CLI source carries separator comments and no license or @file header

Evidence: cli/sushiblas/cli.py:37-39 and 142-144 are `# ----` separator blocks. No file under cli/sushiblas/ opens with a license block, while sushicore/provision/commands.py:1-2 does. Commands in cli.py:48, 77, 111, 127, 137, 150, 161 have no return annotation.

Recommendation: Fold into the header and comment pass for the repository; listed here because it affects the shared CLI shape.

Review: confirmed. cli.py:37-39 and 142-144 are separator blocks; no file under cli/sushiblas/ has a license header (commands.py:1-2 in sushicore does); the seven command functions lack return annotations. The finding stops short of the other python-code-style gaps in the same files (see missed).

#### C22. [low] Python floor is 3.10 while the house style states 3.11

Evidence: cli/pyproject.toml:10 `requires-python = ">=3.10"` and :15 the `tomli` fallback; the python-code-style skill opens with "Python 3.11 or later". sushicore/config.py:37-40 carries the same fallback.

Recommendation: Pick one floor for all CLIs and sushicore; if 3.11, drop the tomli dependency.

Review: confirmed. pyproject.toml:10 '>=3.10' and :15 tomli fallback; python-code-style opens with 'Python 3.11 or later'; sushicore/config.py:37-40 has the fallback.

### Not checked

- Nothing was executed: no `sb` command, no test, no build. Every behaviour claim comes from reading source; F16 in particular is inferred.
- cli/tests/ (test_config, test_suites, test_console_lazy, test_help_screen, test_provision_commands, conftest) were not read.
- sushicore internals read only in part: diag.py, build_env.py, discovery.py top half, console.py, renderer.py, theme.py, help/ and ui/ were not read in full, so the exact output of `sb config` and `sb env` is not described.
- Valid values of `sb setup --toolchain` (sushicore/provision/selection.py COMPONENTS) were not read, so CLI_GUIDE.md:257 `--toolchain oneapi` is unverified.
- sushitrack (`st`) and hub command trees were not compared option by option; se was compared only for version flag, `docs`, `--pick` and panels.
- docs/CONTRIBUTING.md, docs/CLAUDE.md, docs/INTEGRATION.md, docs/GPU_VALIDATION.md and docs/CHANGELOG.md were only grepped for `sb` lines, not read; docs/agent/reports/ were not read.
- Whether `nvptx64-nvidia-cuda` (CLI_GUIDE.md:101) and `nvidia_gpu_sm_86` (cli.py:58) are both accepted SB_SYCL_TARGETS values was not checked against the CMake files.
- The Doxyfile was checked only for OUTPUT_DIRECTORY, HTML_OUTPUT and WARN_LOGFILE, which match CLI_GUIDE.md.
- `hub install-cli` behaviour (pipx, editable, sushicore injection) described in CLI_GUIDE.md:23-32 was not verified against sushihub source.
- Hard-coded paths and platform assumptions: none found in cli/sushiblas/*.py beyond the commented Windows examples in cli/config.toml:33-39; sushicore's own platform handling was not audited.

## Documentation

### Facts

- Tracked documentation is 14 files: NOTICE.md and LICENSE at the root, 11 files flat in docs/, docs/api/mainpage.md, and two files in docs/agent/reports/. Tracked Markdown totals about 3,480 lines.
- EXISTS: docs/README.md (207 lines), docs/CONTRIBUTING.md (404 lines), docs/agent/reports/ (2 files), root LICENSE, root NOTICE.md.
- MISSING: root README.md. The repository has no front door; docs/README.md holds the front-door content instead.
- MISSING: docs/DOCUMENTATION_STYLE_GUIDE.md, docs/getting_started/, docs/architecture/, docs/modules/, docs/guides/, docs/reference/ (so no docs/reference/CHANGELOG.md, GLOSSARY.md or KNOWN_ISSUES.md).
- MISSING: docs/design/ entirely (no README.md topic map, no REMAINING_WORK.md, no design documents), docs/agent/specs, docs/agent/plans, docs/archive/.
- MISSING: every per-module README.md. None exists in cli/, kernels/, src/, include/SushiBLAS/, tests/, examples/ or cmake/ (find for README* outside build/ and docs/api-site/ returns only docs/README.md).
- No place in the architecture: docs/ARCHITECTURE.md (869 lines, loose). Placement question 5, the manual: docs/architecture/, split by its ten numbered sections.
- No place: docs/CLI_GUIDE.md (283 lines, loose). Question 1, a fact about one module: cli/README.md, with a pointer from docs/guides/ if a manual page is kept.
- No place: docs/INTEGRATION.md (170 lines, loose). Question 5: docs/guides/.
- No place: docs/GPU_VALIDATION.md (181 lines, loose). It is a list of unverified claims awaiting a GPU run, so question 3, intent with an open phase: docs/design/ with a status line, its open items listed in REMAINING_WORK.md.
- No place: docs/CHANGELOG.md (784 lines, loose). Belongs at docs/reference/CHANGELOG.md; its 'Known issues' block belongs in docs/reference/KNOWN_ISSUES.md.
- No place: docs/LICENSE, a byte-identical duplicate of the root LICENSE (diff -q reports no difference). Apache-2.0, 190 lines. One copy at the root is enough.
- No place: docs/CLAUDE.md (49 lines), agent instructions stored under docs/. Not a manual page; the conventional home is the repository root. .github/workflows/ci.yml:24 cites it by its current path.
- No place: docs/SECURITY.md and docs/CODE_OF_CONDUCT.md, loose community files. The architecture names no slot; GitHub reads them from the root, docs/ or .github/, so the owner has to decide.
- No place: docs/api/ (mainpage.md, the Doxygen main page, read by Doxyfile lines 34 and 38) and docs/api-site/ (generated Doxygen output, gitignored, present on disk). Neither folder is in the allowed tree; question 5 puts mainpage.md under docs/reference/.
- docs/agent/reports/ holds 2026-09-22-help-screen-task-28.md and 2026-09-22-wave-1-task-7d.md. This matches the global CLAUDE.md layout (specs|plans|reports) but not the documentation skill's layout (docs/agent/<YYYY_MM_DD>_<WORK_NAME>/REPORT.md). The two yardsticks disagree here.
- File names: the two agent reports use lowercase-hyphen names, not UPPER_SNAKE_CASE; the other docs/ files are upper case.
- Version stated in code is 0.1.0 in three places that agree: CMakeLists.txt:15, cli/pyproject.toml:7, include/SushiBLAS/version.hpp:49. git tag lists no tags.
- CLI command set in cli/sushiblas/cli.py: build, test, run, clean, doxygen, config, env, plus setup, doctor, link, unlink registered from sushicore. docs/CLI_GUIDE.md documents all eleven and the options --type, --clean, --define, --suite, --filter, --repeat, --sort, --all match the source.
- A relative-link and anchor check over all tracked Markdown found 5 failures (4 broken paths, 1 broken anchor); all other relative links and anchors resolve.

### Corrections from review

- Tracked documentation is 16 files, not 14: LICENSE and NOTICE.md at the root, 11 flat in docs/, docs/api/mainpage.md and 2 agent reports (git ls-files). The audit's own breakdown adds to 16.
- The README search also returns cli/.pytest_cache/README.md (untracked cache). The conclusion that no module README exists stands.
- F7 cites docs/GPU_VALIDATION.md line 5 for 'Nothing below is a report of a passing test'; the sentence is on line 7. Line 5 holds 'That is not a small caveat'.
- F10 calls the Discord reference unresolvable; docs/SECURITY.md:14-15 does carry the invite, so the pointer resolves. docs/reference/MODULE_MANIFEST.md exists in D:/Projects/sushistack, so sushi-module.toml:3 resolves too.
- F8 lists D1 in two places; it is also at docs/ARCHITECTURE.md:822.

### Findings

#### D1. [high] No root README.md; four links point at it

Evidence: D:/Projects/sushiblas/README.md does not exist (ls of the root shows LICENSE, NOTICE.md, no README). Broken links: docs/ARCHITECTURE.md:8 `../README.md`; docs/ARCHITECTURE.md:868 `../README.md#project-layout`; docs/CLI_GUIDE.md:10 `../README.md`; docs/CLI_GUIDE.md:283 `../README.md#building-with-cmake-directly`. The content those links expect (Project layout, Building with CMake directly) lives in docs/README.md.

Recommendation: Create a root README.md as the front door (what the library is, requirements, quick start, licence, link to docs/README.md) and move the front-door content out of docs/README.md into it. The four links then resolve as written.

Review: confirmed. Root has no README.md (ls). The four links at docs/ARCHITECTURE.md:8, :868 and docs/CLI_GUIDE.md:10, :283 point at ../README.md; the target headings 'Project layout' and 'Building with CMake directly' are in docs/README.md:166 and :85.

#### D2. [high] docs/README.md is a front door, not a manual index; most documents are unreachable from it

Evidence: docs/README.md (207 lines) holds requirements, build, test and layout sections. It links only ARCHITECTURE.md, INTEGRATION.md, ../NOTICE.md and ../LICENSE. Not linked from it: docs/CLI_GUIDE.md, docs/CONTRIBUTING.md, docs/CHANGELOG.md, docs/GPU_VALIDATION.md, docs/SECURITY.md, docs/CODE_OF_CONDUCT.md, docs/CLAUDE.md, docs/api/mainpage.md, docs/agent/reports/2026-09-22-help-screen-task-28.md, docs/agent/reports/2026-09-22-wave-1-task-7d.md.

Recommendation: Rewrite docs/README.md as an index that lists every document by section once the tree is rebuilt; front-door prose moves to the root README.md (F1).

Review: confirmed. Link scan of docs/README.md shows only ARCHITECTURE.md, INTEGRATION.md, ../NOTICE.md, ../LICENSE and external URLs; its headings are Requirements, Quick start, Building, Running the tests, Project layout, so it is a front door and not an index.

#### D3. [high] Documentation architecture is absent: flat docs/ with no manual folders, design, archive or reference

Evidence: docs/ contains 11 loose files plus api/, api-site/ and agent/reports/. Missing: docs/DOCUMENTATION_STYLE_GUIDE.md, docs/getting_started/, docs/architecture/, docs/modules/, docs/guides/, docs/reference/ (CHANGELOG.md, GLOSSARY.md, KNOWN_ISSUES.md), docs/design/ (README.md, REMAINING_WORK.md), docs/agent/specs, docs/agent/plans, docs/archive/.

Recommendation: Per the global rule, report the gaps and get approval before building the skeleton as its own task. Destinations for each loose file are listed in the facts.

Review: confirmed. ls docs shows 11 loose files plus agent/reports, api, api-site; none of the manual folders, design/, archive/, reference/ or DOCUMENTATION_STYLE_GUIDE.md exists.

#### D4. [high] Changelog breaks the one-line rule almost everywhere

Evidence: docs/CHANGELOG.md is 784 lines. Of 70 top-level entries only 7 match `- YYYY-MM-DD — ...` (lines 19-30). There are 31 nested bullets. Lines 54-767 are multi-paragraph entries that explain reasons, for example the `convert` entry near line 190 ('The naive spelling ... is 7 x 7 = 49 kernel instantiations') and the count_non_finite entry before it. Sections are Keep-a-Changelog style (`## [Unreleased]`, `### Build`, `### CLI`, `### Added`, `### Documented`, `### Changed` twice at lines 103 and 510) instead of `## Unreleased` with scope-prefixed lines. Only line 26 carries a scope (`cli:`). A template block `## [x.y.z] — YYYY-MM-DD` sits at line 769. The file lives at docs/CHANGELOG.md, not docs/reference/CHANGELOG.md.

Recommendation: Move to docs/reference/CHANGELOG.md. Move the reasoning in lines 54-767 into design documents (or docs/archive/ if the work is closed) and replace each entry with one dated, scoped, past-tense line under 240 characters. Drop the `###` category headings and the template block.

Review: confirmed. Recomputed: 784 lines, 70 top-level bullets, 31 nested, 7 dated entries (lines 19-30), only line 26 has a scope; headings are '## [Unreleased]', '### Changed' at 103 and 510, template at 769. The 7 dated lines are 87-173 characters, so they pass the length rule.

#### D5. [medium] CONTRIBUTING forbids changelog entries that the changelog nonetheless carries, and never mentions the required doc architecture

Evidence: docs/CONTRIBUTING.md:302-306: 'Do not add entries to CHANGELOG.md until the project cuts its v1.0.0 release'. docs/CHANGELOG.md:19-30 has dated entries up to 2026-10-04. docs/CHANGELOG.md:7 says 'The project has not yet been assigned a semantic version', while CMakeLists.txt:15, cli/pyproject.toml:7 and include/SushiBLAS/version.hpp:49 all say 0.1.0 and docs/INTEGRATION.md:81-83 documents `find_package(SushiBLAS 0.1 REQUIRED)`. CONTRIBUTING section 6 names no module README, design status, REMAINING_WORK or agent work folder.

Recommendation: Rewrite CONTRIBUTING section 6 to state the same-commit rule in its current form (changelog line, module README, design status) and remove the 'no version yet' statements from both files.

Review: confirmed. docs/CONTRIBUTING.md:302-306 forbids entries; docs/CHANGELOG.md:19-30 has them; CHANGELOG.md:7 says no version while CMakeLists.txt:15, cli/pyproject.toml:7, version.hpp:49 say 0.1.0 and INTEGRATION.md:82 uses find_package(SushiBLAS 0.1).

#### D6. [medium] CONTRIBUTING's source-comment and licence-header rules contradict the global standard

Evidence: docs/CONTRIBUTING.md:98-123 prescribes a banner header made of separator lines with an Apache-2.0 block; the global rule forbids separator lines and requires `/** @file` with `@brief` and `@author`. docs/CONTRIBUTING.md:188-196 prescribes `// TODO:` comments in source; the global rule forbids TODO in source. docs/CONTRIBUTING.md:185-186 asks for Doxygen only 'on every public API'; the rule covers every symbol, private members included. docs/CONTRIBUTING.md:10-11 fixes contributions to Apache-2.0, which the pending licence decision will change.

Recommendation: Replace the style sections with a pointer to the shared rules (source-comments, cpp-code-style) and update the licence paragraph and header template once the new licence is chosen.

Review: confirmed. Read docs/CONTRIBUTING.md:96-123 (separator banner with Apache block), :183-186 (Doxygen on public API only), :188-196 (// TODO endorsed), :10-11 (Apache-2.0). All conflict with the global source-comment rule.

#### D7. [medium] More than one backlog, none of them REMAINING_WORK.md

Evidence: No docs/design/REMAINING_WORK.md. Open work is recorded in docs/GPU_VALIDATION.md (181 lines; line 5: 'Nothing below is a report of a passing test. It is a list of claims awaiting one.'), in docs/CHANGELOG.md:32-33 ('### Known issues' with a '#### Closed' sub-list), and as source TODOs that docs/CONTRIBUTING.md:188-196 endorses and docs/ARCHITECTURE.md:829 cites ('still a `// TODO:` stub').

Recommendation: Create docs/design/REMAINING_WORK.md as the single backlog; move GPU validation items and the sparse stub into it, and known issues into docs/reference/KNOWN_ISSUES.md.

Review: confirmed. No docs/design/REMAINING_WORK.md; GPU_VALIDATION.md:7 has the quoted sentence (audit says line 5), CHANGELOG.md:32-35 has Known issues/Closed, ARCHITECTURE.md:829 cites the TODO stub.

#### D8. [medium] No design documents and no status lines; design intent is scattered

Evidence: docs/design/ does not exist. Decisions are cited by bare identifiers with no document behind them: 'D3' (docs/README.md:17, docs/INTEGRATION.md:75, docs/ARCHITECTURE.md:793), 'D1' (docs/CLI_GUIDE.md:223, docs/api/mainpage.md), 'SB-6' (docs/CONTRIBUTING.md:223, docs/INTEGRATION.md:170). docs/GPU_VALIDATION.md is open intent without a `**Status:**` line. The reasoning lives in changelog paragraphs (F4).

Recommendation: Open docs/design/ with a README.md topic map; give D1, D3, SB-6 and GPU validation one document each with a status on line 2, fed from the changelog paragraphs.

Review: confirmed. grep finds D3 at README.md:17, INTEGRATION.md:75, ARCHITECTURE.md:793; D1 at CLI_GUIDE.md:223, ARCHITECTURE.md:822, api/mainpage.md:28; SB-6 at CONTRIBUTING.md:223, INTEGRATION.md:170. No docs/design/, GPU_VALIDATION.md has no status line.

#### D9. [medium] docs/README.md contradicts the code on dispatch helpers, accessors and layout

Evidence: docs/README.md:175-176 places `dispatch_with_mode` in `engine/blas/utils.hpp`; it lives in include/SushiBLAS/engine/task.hpp (utils.hpp:110 says it 'used to live here and now lives in' task.hpp; docs/ARCHITECTURE.md:77 agrees with the code). docs/README.md:150 and :172 list seven Engine accessors and omit `spatial()`, which exists at include/SushiBLAS/engine.hpp:134. The layout block omits include/SushiBLAS/engine/spatial/, spatial.hpp, task.hpp, dispatch.hpp, launch_1d.hpp, reduce_1d.hpp, reduce_axis.hpp, device_policy.hpp, core/export.hpp, core/traits.hpp, cmake/, sushi-module.toml and Doxyfile.

Recommendation: Correct the helper location, add `spatial()`, and regenerate the layout block from the tree (it belongs in the root README or docs/architecture/).

Review: confirmed. dispatch_with_mode is defined at include/SushiBLAS/engine/task.hpp:166, utils.hpp:110 says it moved; README.md:175-176 still places it in utils.hpp. engine.hpp has spatial(); README.md:150 and :171-172 omit it. The layout block omits engine/spatial, task.hpp, dispatch.hpp, launch_1d.hpp, reduce_*.hpp, device_policy.hpp, core/export.hpp, core/traits.hpp, cmake/, sushi-module.toml, Doxyfile.

#### D10. [medium] Broken anchor and unresolvable references

Evidence: docs/ARCHITECTURE.md:209 links `#6-dispatch-the-generic-op-seam`; the heading at line 299 is '## 6. The dispatch helpers (`engine/task.hpp`)', so the anchor does not exist. docs/CONTRIBUTING.md:390 says 'Use the GitHub issue templates'; .github/ contains only workflows/ci.yml. docs/CONTRIBUTING.md:399-400 sends readers to SECURITY.md 'for the current invite' to a Discord that is for private vulnerability contact there (docs/SECURITY.md:14-15). sushi-module.toml:3 cites `docs/reference/MODULE_MANIFEST.md in the SushiStack repository`, not verified here.

Recommendation: Fix the anchor to the real heading slug, either add issue templates or drop the sentence, and point the Discord reference at CODE_OF_CONDUCT.md:50 or a contact section.

Review: confirmed. Anchor at ARCHITECTURE.md:209 does not match the heading at :299; .github holds only workflows/. Two sub-items are weak: SECURITY.md:14-15 does carry the invite, so CONTRIBUTING.md:399 resolves, and docs/reference/MODULE_MANIFEST.md exists in D:/Projects/sushistack.

#### D11. [medium] docs/CLAUDE.md is misplaced, off-register and conflicts with the global rules

Evidence: docs/CLAUDE.md sits under docs/ (the architecture has no slot). Line 8 tells the agent to 'read every `.md` file under `docs/`', about 3,480 lines including a 784-line changelog. Line 49 ends 'love you boss, kolay gelsin.', Turkish in an English artifact. Line 7 cites `docs/CONTRIBUTING.md` whose style rules conflict with the global ones (F6). .github/workflows/ci.yml:24 cites the file by path.

Recommendation: Move to the repository root, cut it to project-specific facts (the `sb` build rule, layer boundaries), remove what the global CLAUDE.md already states, and update the ci.yml comment.

Review: confirmed. Read docs/CLAUDE.md in full: line 8 'read every .md file under docs/', line 49 'love you boss, kolay gelsin.'; ci.yml:24 cites docs/CLAUDE.md. Moving it to the root is an owner decision, as the audit's own fact list says.

#### D12. [medium] Prose in the manual breaks the humanizer register and needs a rewrite

Evidence: Em dash as a tic and corrective contrast throughout: docs/README.md:3-10 has seven em dashes in four bullets and 'It never silently does nothing', 'an honest map of its edges'; docs/INTEGRATION.md:1-41 ('The fact that shapes everything', 'Obviously.' twice at lines 93-94, 'the compile-time macro is the whole story'); docs/CLI_GUIDE.md:211-214 narrates history ('for a while it was not, and `.gitignore` listed it'), which belongs in a design document; docs/GPU_VALIDATION.md:3-5 ('That is not a small caveat, and it is the reason this file exists'); docs/CHANGELOG.md:54-767 is essay prose with bold lead-ins and em dashes in nearly every bullet; docs/api/mainpage.md closes with 'That bug has cost this stack two silently wrong training runs'. Paragraphs of 80-120 words in a single sentence are common in docs/README.md:5-10.

Recommendation: Rewrite docs/README.md, INTEGRATION.md, GPU_VALIDATION.md and the changelog through the humanizer pass when they are moved; CLI_GUIDE.md and CONTRIBUTING.md need lighter cuts (history and rationale out, em dashes down).

Review: confirmed. Quotes found at INTEGRATION.md:8, :93-94, GPU_VALIDATION.md:5, CLI_GUIDE.md:211-214, mainpage.md closing line. Em dash counts: README 21 in 1761 words, GPU_VALIDATION 32 in 2398, CHANGELOG 84, ARCHITECTURE 113 in 7136. ARCHITECTURE.md has the same density and is missing from the rewrite list.

#### D13. [medium] Agent report claims a file that does not exist

Evidence: docs/agent/reports/2026-09-22-wave-1-task-7d.md:9 says it 'Added one line to `docs/reference/CHANGELOG.md`, creating the file', and lines 70-83 say the repository now has two changelogs. docs/reference/ does not exist; the entry sits at docs/CHANGELOG.md:21. Line 12 says 'No commit was made' while commit e7783de is on main. The audit listed this report as not read.

Recommendation: When the report moves into its work folder, correct the path and the commit state in place, or archive it as it is with the mismatch recorded in the close-out commit.

Review: found by review.

#### D14. [medium] Manual pages disagree on whether CMake may be run directly

Evidence: docs/api/mainpage.md:14 describes CONTRIBUTING as 'the build rules (`sb build`, never cmake directly)' and docs/CLAUDE.md forbids invoking cmake or ninja. docs/README.md:85 has a section 'Building with CMake directly', docs/README.md:124-127 gives ctest commands, and docs/CLI_GUIDE.md:279-283 says the CLI is 'a convenience, not a requirement'.

Recommendation: State the rule once in CONTRIBUTING (who must use `sb`, who may call CMake) and make mainpage.md, the README and CLI_GUIDE.md cite that one sentence.

Review: found by review.

#### D15. [medium] ARCHITECTURE.md is left out of the humanizer rewrite list

Evidence: docs/ARCHITECTURE.md has 113 em dashes in 7,136 words (one per 63 words, against the skill's one per 300), the highest count in the manual. F15 names README, INTEGRATION, GPU_VALIDATION, CHANGELOG, CLI_GUIDE and CONTRIBUTING but not this file, and the audit read only about 20 of its 869 lines.

Recommendation: Add docs/ARCHITECTURE.md to the rewrite scope and run the humanizer pass when it is split into docs/architecture/.

Review: found by review.

#### D16. [low] Duplicate LICENSE under docs/

Evidence: docs/LICENSE and LICENSE are identical (diff -q silent), 190 lines each, Apache-2.0. docs/README.md:207 and docs/CONTRIBUTING.md:11 already link the root copy.

Recommendation: Delete docs/LICENSE (needs owner approval) so the coming licence change touches one file.

Review: confirmed. diff -q LICENSE docs/LICENSE returned 0 with no output; both 190 lines; docs/README.md:207 and CONTRIBUTING.md:11 link ../LICENSE.

#### D17. [low] Module facts live in the manual; no module README exists

Evidence: docs/CLI_GUIDE.md (283 lines) is entirely about cli/; there is no cli/README.md (docs/agent/reports/2026-09-22-help-screen-task-28.md notes 'The CLI has no `cli/README.md`'). Kernel rules for kernels/ and src/engine/ sit in docs/CONTRIBUTING.md:231-294 and docs/ARCHITECTURE.md:366-792; tests/package/ is described only in docs/INTEGRATION.md:68-70 and docs/CLI_GUIDE.md:158-182.

Recommendation: Add README.md to cli/, kernels/, src/, include/SushiBLAS/, tests/ and examples/, moving each module's facts beside its code.

Review: confirmed. No README in cli/, kernels/, src/, include/SushiBLAS/, tests/, examples/, cmake/; CLI_GUIDE.md is all about cli/; the help-screen report line 8 has the quoted sentence. One README per module is a brick-shaped fix.

#### D18. [low] Agent reports use the wrong names and one records an unfinished state

Evidence: docs/agent/reports/2026-09-22-help-screen-task-28.md and 2026-09-22-wave-1-task-7d.md use lowercase-hyphen names; the skill requires `docs/agent/<YYYY_MM_DD>_<WORK_NAME>/REPORT.md`. The first states 'Branch `feature/sushicore_help_screen` ... Not pushed, merged or tagged. The checkout is left on that branch', while the work is on main and in the changelog (docs/CHANGELOG.md:27). It also says '`sb` has 7 commands'; the CLI now has eleven.

Recommendation: Decide which agent layout is canonical (global CLAUDE.md and the documentation skill differ), rename accordingly, and move both closed reports to docs/archive/ once they pass 90 days untouched.

Review: confirmed. Both report names are lowercase-hyphen; report line 3 says the branch is unmerged while commit f77a723 is on main; line 7 says 7 commands, cli.py registers 7 plus the four from sushicore. The local branch feature/sushicore_help_screen still exists.

#### D19. [low] Doxygen main page and generated site sit in folders the tree does not allow

Evidence: docs/api/mainpage.md is read by Doxyfile:34 and :38; output goes to docs/api-site (Doxyfile:24), gitignored at .gitignore `docs/api-site/html/`. docs/api/mainpage.md lists four prose documents by path and omits CLI_GUIDE.md.

Recommendation: Move the main page to docs/reference/ and the generated output outside docs/ (for example build/), updating Doxyfile, .gitignore, docs/CLI_GUIDE.md:66 and :208 and docs/README.md in the same change. Doxyfile is build-adjacent configuration, so this needs owner approval.

Review: confirmed. Doxyfile:24 OUTPUT_DIRECTORY docs/api-site, :34 and :38 read docs/api/mainpage.md; .gitignore ignores docs/api-site/html/ and the warnings log; mainpage.md lists README, ARCHITECTURE, INTEGRATION, CONTRIBUTING and omits CLI_GUIDE.md.

### Not checked

- docs/ARCHITECTURE.md body (869 lines): only headings, lines 1-10, 205-211 and 866-869 were read. Its claim that 'every class and file named here exists' was not verified symbol by symbol.
- docs/CHANGELOG.md lines 40-767 were sampled (about 100 lines read plus heading and count statistics), not read in full; symbol and path citations inside were not verified.
- docs/GPU_VALIDATION.md beyond line 25, docs/SECURITY.md and docs/CODE_OF_CONDUCT.md bodies (only contact lines read).
- docs/agent/reports/2026-09-22-wave-1-task-7d.md was not read; only the first 20 lines of the help-screen report were.
- NOTICE.md content (attribution accuracy for the portBLAS-derived code) was not reviewed; only its links were checked.
- External URLs (sushisystems.io install scripts, GitHub links, the Discord invite, SushiRuntime INTEGRATION.md) were not fetched.
- Cross-repository claims: whether `hub install-cli`, `hub link`, `hub status` exist in sushistack, and whether docs/reference/MODULE_MANIFEST.md exists there.
- Behavioural claims that need running the CLI or a build (exit codes, `--suite package` behaviour, `sb doxygen` output, SB_* CMake option defaults in docs/CLI_GUIDE.md:109-114) were not verified; only option names were compared against cli/sushiblas/cli.py.
- Inline backticked path citations in prose (as opposed to Markdown links) were not checked systematically.
- docs/api-site/ generated HTML, build/, and cli/sushiblas_cli.egg-info/ were skipped as instructed.
- Source-comment and licence-header compliance of the code itself (a different dimension).

## Layout and hygiene

### Facts

- Root of D:/Projects/sushiblas: .github/ (one workflow), .gitignore, build/ (ignored CMake tree, 206 MB), cli/ (Python CLI package `sushiblas-cli`, alias `sb`), cmake/ (10 CMake modules), CMakeLists.txt, CMakePresets.json, docs/, Doxyfile, examples/ (2 .cpp + CMakeLists), include/SushiBLAS/ (public headers), kernels/ (38 tracked headers: blas.hpp, math.hpp, random.hpp, spatial.hpp, reference/, sycl/), LICENSE (Apache-2.0), NOTICE.md, src/ (115 files, engine/ + engine.cpp + io.cpp), sushi-module.toml, tests/ (common, functional/{unit,integration,regression}, package).
- Absent from the root: README.md, AGENTS.md, CLAUDE.md, .clang-format, .editorconfig, .gitattributes, tools/, third_party/.
- 361 files tracked. Per top-level: src 115, tests 109, include 44, kernels 38, cli 20, docs 14, cmake 10, examples 3, .github 1, plus 7 root files.
- Versions: CMakeLists.txt:15 `project(SushiBLAS VERSION 0.1.0)`; include/SushiBLAS/version.hpp:49 `SUSHIBLAS_VERSION_STRING "0.1.0"` (hand-written, checked against CMake at CMakeLists.txt:66-76); cli/pyproject.toml:7 `version = "0.1.0"`. These three agree.
- Versions that do not carry 0.1.0: Doxyfile:22 `PROJECT_NUMBER = "v1 (unreleased)"`; docs/CHANGELOG.md:7-8 says the project 'has not yet been assigned a semantic version'; sushi-module.toml declares no version field.
- Git tags: none. 69 commits on main. No stash, one worktree.
- Remote: origin https://github.com/SushiSystems/SushiBLAS.git. sushi-module.toml gives repo = https://github.com/sushisystems/sushiblas.git (same repository, different case).
- Branches: main, feature/blas-spatial-domain, feature/sushicore_help_screen (local only); both feature branches have 0 commits ahead of main. main is 1 commit ahead of origin/main (aa5633e, unpushed).
- git status: clean, nothing untracked. Ignored and present on disk: build/, cli/.pytest_cache/, cli/sushiblas/__pycache__/, cli/sushiblas/services/__pycache__/, cli/tests/__pycache__/, cli/sushiblas_cli.egg-info/, docs/api-site/ (247 HTML files, 4.0 MB, plus doxygen-warnings.log).
- .gitignore (36 lines): [Bb]uild/, bin/, out/; .vscode/, .vs/, *.swp, *.swo; CMakeFiles/, CMakeCache.txt, cmake_install.cmake, Makefile, install_manifest.txt, compile_commands.json, CTestTestfile.cmake, _deps/; docs/api-site/html/, docs/api-site/doxygen-warnings.log, html/, latex/; cli/config.local.toml, __pycache__/, *.pyc, cli/*.egg-info/, .sushi_env.json.
- CI: one workflow, .github/workflows/ci.yml, triggers push to main and pull_request. Job `functional` (ubuntu, intel/oneapi-basekit container, direct cmake + ctest). Job `cli` (ubuntu, Python 3.11, pyflakes with `|| true`, pytest). No Windows job, no tag-triggered release workflow.
- tools/: does not exist. None of the four standard checkers is present.
- docs/ tracked: ARCHITECTURE.md, CHANGELOG.md, CLAUDE.md, CLI_GUIDE.md, CODE_OF_CONDUCT.md, CONTRIBUTING.md, GPU_VALIDATION.md, INTEGRATION.md, LICENSE, README.md, SECURITY.md, api/mainpage.md, agent/reports/ (2 reports). No getting_started/, architecture/, modules/, guides/, reference/, design/, archive/.
- docs/CHANGELOG.md is 784 lines with a single `## [Unreleased]` section (line 15) and a template heading `## [x.y.z] — YYYY-MM-DD` at line 769; no released section.
- No tracked binaries, images, logs or generated files. The only tracked .json is CMakePresets.json. Largest tracked file is docs/ARCHITECTURE.md at 54 KB. .git is 2.9 MB.
- No loose images, compiler probe headers (CXX-DetectStdlib-*.h), imgui.ini or cmake discovery JSON in this repository's root; sibling sushiruntime does carry four CXX-DetectStdlib-*.h files in its root, sushiblas does not.
- cli/sushiblas_cli.egg-info/PKG-INFO (ignored, on disk) records `Requires-Dist: sushicore>=0.1.0` while cli/pyproject.toml:12 requires `sushicore>=0.7.0`; the editable-install metadata is stale.
- Sibling comparison: sushiruntime has tools/ (packaging only), src/, Doxyfile in root and no root README, the same shape as sushiblas. sushiengine has tools/documentation and tools/layering, root README.md and CLAUDE.md. sushistack and sushicore have ci.yml plus release.yml.

### Corrections from review

- F1 recommendation and the sibling fact present sushiengine as the model for the four checkers. D:/Projects/sushiengine/tools carries only check_source_comments.py of the four; it has check_include_layering.py instead of check_layering.py and no check_docs_layout.py or check_changelog.py. The four under the standard names exist in D:/Projects/sushiskills/tools (and in sushiai, sushidsp, sushitrack); those are the copies to model on.
- Facts line 'Branches: main, feature/blas-spatial-domain, feature/sushicore_help_screen (local only)' reads as if both feature branches were local only. feature/blas-spatial-domain tracks origin/feature/blas-spatial-domain at c420bb1; only feature/sushicore_help_screen has no remote. F14 states it correctly.
- .gitignore is 35 lines, not 36 (the last line, .sushi_env.json, has no trailing newline). No conclusion changes.
- F12 cites docs/CHANGELOG.md:121-125 as the paragraph entry; the entry starts at 121 and continues beyond line 130.
- F9 lists sushi-module.toml:3 as a wrong-place citation. The comment names the SushiStack repository explicitly and the file exists at D:/Projects/sushistack/docs/reference/MODULE_MANIFEST.md, so it resolves.
- F8 says the layout's fix may be to fold kernels/ into include/SushiBLAS/kernels/. kernels/ is a private include root (src/CMakeLists.txt:163-170, 'never installed and never PUBLIC'), included by 40 files under src/, 5 under kernels/ and 1 under tests/, and by no public header.
- Recomputed and correct: 361 tracked files with the per-folder split (src 115, tests 109, include 44, kernels 38, cli 20, docs 14, cmake 10, examples 3, .github 1, root 7); 69 commits; no tags; versions 0.1.0 at CMakeLists.txt:15, version.hpp:49, cli/pyproject.toml:7; changelog 784 lines with headings at 15 and 769; LICENSE copies identical.

### Findings

#### L1. [high] No tools/ folder; all four mandatory checkers are missing

Evidence: D:/Projects/sushiblas/tools does not exist (`ls -R tools` returns nothing; `git ls-files` lists no tools/ path). project-tools requires tools/documentation/check_source_comments.py, check_docs_layout.py, check_changelog.py and tools/layering/check_layering.py. Sibling D:/Projects/sushiengine/tools carries documentation/ and layering/.

Recommendation: Add tools/documentation/ and tools/layering/ with the four checkers under the standard names and contract, modelled on sushiengine's copies.

Review: confirmed. D:/Projects/sushiblas has no tools/ (ls -A of root, git ls-files). The defect is real, but the recommendation's model is wrong: sushiengine carries only check_source_comments.py of the four; see wrong_facts.

#### L2. [high] CI `cli` job installs sushicore from a path that no longer exists

Evidence: .github/workflows/ci.yml, `cli` job: checks out SushiSystems/SushiStack into .sushistack and runs `python -m pip install -e ./.sushistack/sushicore`, with the comment 'sushicore is published to no index, so it cannot be a pip dependency'. The local D:/Projects/sushistack checkout has no sushicore/ directory (entries: cli contract dependencies docs gui install.ps1 install.sh LICENSE README.md tools); sushicore is its own repository at D:/Projects/sushicore. docs/CHANGELOG.md:21 records 'Took `sushicore` from PyPI instead of a sibling checkout' and cli/pyproject.toml:12 requires sushicore>=0.7.0.

Recommendation: Replace the SushiStack checkout and the editable install with `pip install -e cli` (which pulls sushicore from PyPI), and delete the stale comment. Confirm against the remote SushiStack main, which this audit could not read.

Review: confirmed. .github/workflows/ci.yml:140-153 installs ./.sushistack/sushicore; `git -C D:/Projects/sushistack ls-tree origin/main` lists no sushicore/ either, so the remote-tracking ref agrees with the working tree. Caveat on the fix: sushicore is tagged only to v0.4.0 while cli/pyproject.toml:12 needs >=0.7.0 (see missed).

#### L3. [high] CI departs from the house shape: one workflow, no Windows, no checkers, no release trigger, direct cmake/ctest

Evidence: .github/workflows/ci.yml is the only workflow; both jobs are `runs-on: ubuntu-latest`. The `functional` job runs `cmake -B build ...`, `cmake --build build`, `ctest --test-dir build ...` directly and the `cli` job runs `python -m pytest` directly; the file's own header ('Why this file calls cmake/ctest directly instead of `sb`') admits the departure. No job runs a checker, no `vX.Y.Z` tag trigger exists. continuous-integration requires push.yml and release.yml, Windows and Linux on every trigger, and CLI-only invocation. Siblings sushistack and sushicore have ci.yml + release.yml.

Recommendation: Split into push.yml and release.yml, add a Windows lane, run the four checkers, and route build and test through `sb`. The reason `sb` cannot run on a stock runner is an owner decision to settle (provision in CI, or record the exception in a design document instead of a 50-line YAML comment).

Review: confirmed. ci.yml is the only workflow, both jobs ubuntu-latest, cmake/ctest at lines 118-127 and pytest at 162 called directly, no tag trigger, no checker job. Note that no sibling has push.yml either (sushistack and sushicore use ci.yml + release.yml, sushiruntime and sushiengine ci.yml only), so the rename is stack-wide. The finding bundles five separate defects into one.

#### L4. [medium] pyflakes step in CI can never fail

Evidence: .github/workflows/ci.yml, step 'Import-safety check': `run: python -m pyflakes sushiblas || true`. continuous-integration: 'A job is never disabled or marked allowed-to-fail to get green.'

Recommendation: Drop `|| true`, or remove the step if its findings are not meant to gate.

Review: confirmed. ci.yml:158 `run: python -m pyflakes sushiblas || true`.

#### L5. [medium] ci.yml carries stale statements about the repository

Evidence: .github/workflows/ci.yml, 'Deliberately absent' block: '`sb doxygen` exists and reads a repo-root `Doxyfile`, but this repo has none — and `Doxyfile` is listed in .gitignore'. D:/Projects/sushiblas/Doxyfile is tracked, and .gitignore:21-24 says 'Doxyfile itself is COMMITTED'. The same file refers to `ss install` although docs/CHANGELOG.md:29 records the command renamed to `hub`.

Recommendation: Correct or remove the paragraphs when the workflow is reworked (F3). The explanatory prose belongs in a design document, not in the workflow file.

Review: confirmed. ci.yml:186-189 says the repo has no Doxyfile and that .gitignore lists it; Doxyfile is tracked and .gitignore:22-24 says it is committed. ci.yml:30 still says `ss install`.

#### L6. [medium] Root has no README.md, AGENTS.md, CLAUDE.md, .clang-format, .editorconfig or .gitattributes

Evidence: `ls README.md AGENTS.md CLAUDE.md .clang-format .editorconfig .gitattributes` in D:/Projects/sushiblas: every one 'No such file or directory'. The front door is docs/README.md and the agent instructions are docs/CLAUDE.md, where Claude Code does not auto-load them. cli/pyproject.toml declares no `readme`. With no .gitattributes on a Windows-developed, Linux-CI repository, line endings are left to each machine's git config.

Recommendation: Add a root README.md (what it is, one build line, link to docs/README.md), move docs/CLAUDE.md to the root with an AGENTS.md naming the skills, and add .clang-format, .editorconfig and .gitattributes matching the sibling repositories.

Review: confirmed. Root listing shows none of the six files; docs/CLAUDE.md and docs/README.md exist; cli/pyproject.toml has no `readme` key.

#### L7. [medium] Doxyfile sits in the root, where the layout forbids it

Evidence: D:/Projects/sushiblas/Doxyfile is tracked. repository-layout lists `Doxyfile` among entries forbidden in the root. .gitignore:21-24 explains that `sb doxygen` reads a repo-root Doxyfile, so the CLI hard-codes the location (sushiruntime does the same).

Recommendation: Move it under docs/ (for example docs/api/Doxyfile) and point `sb doxygen` at it, in step with sushiruntime so the two stay the same shape. This touches the CLI contract; ask the owner first.

Review: confirmed. Doxyfile is tracked in the root and repository-layout names it as forbidden there; .gitignore:22-24 documents that `sb doxygen` reads the root one.

#### L8. [medium] Source tree does not match the single-module layout: src/ instead of source/, and a top-level kernels/ folder

Evidence: D:/Projects/sushiblas/src (115 tracked files) where repository-layout names `source/<area>/<name>.cpp`. D:/Projects/sushiblas/kernels (38 tracked headers: blas.hpp, math.hpp, random.hpp, spatial.hpp, reference/, sycl/) is a top-level folder outside the permitted set, while include/SushiBLAS/kernels/ also exists (for example include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp), so kernel headers live in two trees. examples/ is likewise outside the permitted set (the layout's home for executables is applications/).

Recommendation: Decide with the owner: rename src/ to source/, fold kernels/ into include/SushiBLAS/kernels/ or source/, and move examples/ to applications/. sushiruntime uses src/ and examples/ too, so the rename should land in both or the skill should be amended.

Review: confirmed. src/ holds 114 .cpp + CMakeLists.txt, kernels/ 38 headers, include/SushiBLAS/kernels/ 4 headers. One option in the recommendation is not brick-shaped: src/CMakeLists.txt:163-170 keeps kernels/ PRIVATE and uninstalled, so folding it into include/SushiBLAS/kernels/ would publish private headers. The home is source/.

#### L9. [medium] docs/ lacks the standard architecture and holds files in the wrong places

Evidence: D:/Projects/sushiblas/docs holds ARCHITECTURE.md, CHANGELOG.md, CLI_GUIDE.md, GPU_VALIDATION.md, INTEGRATION.md, SECURITY.md, CODE_OF_CONDUCT.md flat at its top. Missing: docs/DOCUMENTATION_STYLE_GUIDE.md, docs/getting_started/, architecture/, modules/, guides/, reference/ (CHANGELOG.md, GLOSSARY.md, KNOWN_ISSUES.md), design/REMAINING_WORK.md, archive/. docs/agent/ has reports/ only, no specs/ or plans/. sushi-module.toml:3 cites 'docs/reference/MODULE_MANIFEST.md' as the format's home, which is in SushiStack, not here.

Recommendation: Build the skeleton as its own approved task: CHANGELOG.md to docs/reference/, ARCHITECTURE.md to docs/architecture/, CLI_GUIDE.md and INTEGRATION.md to docs/guides/, add the missing reference and design files. Detailed document review belongs to the documentation dimension.

Review: confirmed. git ls-files docs matches the list; none of the standard subfolders exist. One clause is not a defect: sushi-module.toml:3 says the format document is 'in the SushiStack repository', and D:/Projects/sushistack/docs/reference/MODULE_MANIFEST.md exists.

#### L10. [medium] docs/LICENSE is a tracked byte-identical copy of the root LICENSE

Evidence: `cmp LICENSE docs/LICENSE` reports identical; both 10777 bytes, Apache License 2.0. Two tracked copies mean the pending licence change has to be made twice and can drift.

Recommendation: Delete docs/LICENSE and link to the root file. It is a deletion of an existing file, so it needs the owner's approval, and it should land with the licence change.

Review: confirmed. cmp LICENSE docs/LICENSE reports identical; both 10777 bytes, both tracked.

#### L11. [medium] No release has ever been tagged and the version statements contradict each other

Evidence: `git tag -l` is empty across 69 commits. CMakeLists.txt:15, include/SushiBLAS/version.hpp:49 and cli/pyproject.toml:7 all say 0.1.0, but Doxyfile:22 says `PROJECT_NUMBER = "v1 (unreleased)"` and docs/CHANGELOG.md:7-10 says the project 'has not yet been assigned a semantic version' and that entries 'begin at the v1.0.0 cut'. docs/CHANGELOG.md has one 784-line `## [Unreleased]` section holding many `feat` and `fix` entries, far past the 15-entry release trigger in versioning-and-release.

Recommendation: Set Doxyfile PROJECT_NUMBER from the CMake version, rewrite the changelog preamble to state 0.1.0, and propose a first tag (v0.1.0) to the owner with the Unreleased section cut under it.

Review: confirmed. No tags, 69 commits; Doxyfile:22 `v1 (unreleased)`; docs/CHANGELOG.md:7-9 says no semantic version assigned. The Unreleased section holds 70 top-level `- ` entries against the 15-entry trigger.

#### L12. [medium] Changelog shape breaks the one-line rule and the heading format

Evidence: docs/CHANGELOG.md:121-125 is a multi-line bolded paragraph entry ('- **`sb doxygen` works.** A repo-root `Doxyfile` and `docs/api/mainpage.md` are / now committed, and `.gitignore` no longer swallows the Doxyfile — it listed ...') that explains why. The file is 784 lines for one section. Headings use Keep-a-Changelog `## [Unreleased]` and `## [x.y.z] — YYYY-MM-DD` (lines 15, 769) where versioning-and-release specifies `## Unreleased` and `## vX.Y.Z — YYYY-MM-DD`. Line 25 prefixes its entry with 'cli:', unlike its siblings.

Recommendation: Rewrite the older entries as one-line dated entries, move the reasons into design documents, and switch the headings to the house form. The count of offending entries needs check_changelog.py (F1).

Review: confirmed. docs/CHANGELOG.md:121 onward is a multi-line bolded paragraph entry (it runs past line 130, not to 125); headings at 15 and 769 use the bracket form; line 25 carries the `cli:` prefix.

#### L13. [medium] The CLI requires a sushicore version that has no tag, and the bump has no changelog entry

Evidence: D:/Projects/sushiblas/cli/pyproject.toml:12 requires `sushicore>=0.7.0`. D:/Projects/sushicore/pyproject.toml:7 says 0.7.0 but `git -C D:/Projects/sushicore tag -l` ends at v0.4.0. D:/Projects/sushiblas/docs/CHANGELOG.md:20 records only the raise to 0.4.0; the raise to 0.7.0 came in aa5633e (2026-10-04) and the entry at line 25 does not name cli/pyproject.toml or the new bound. F2's fix (`pip install -e cli` pulling from PyPI) works only if 0.7.0 is published, which was not checked here (no network call).

Recommendation: Confirm which sushicore versions are on PyPI before changing the CI job; release and tag sushicore 0.7.0 first if it is not there. Add the missing one-line changelog entry for the bound.

Review: found by review.

#### L14. [medium] tests/ does not match the single-module shape

Evidence: D:/Projects/sushiblas/tests holds common/ (4 files), functional/unit (98), functional/integration (2), functional/regression (2) and package/ (2). repository-layout names tests/{unit,integration,regression,common}; functional/ is an extra level over 102 of 109 files and package/ is outside the set. The audit left this under not_checked instead of ruling on it. sushiruntime's tests/ has the same functional/ and package/ folders.

Recommendation: Decide it together with F8, in both repositories at once: drop the functional/ level, and either give tests/package/ a named home or amend the skill to allow it.

Review: found by review.

#### L15. [medium] The stale sushistack/sushicore path also lives in the CLI test fixture and in sushiruntime's CI

Evidence: D:/Projects/sushiblas/cli/tests/conftest.py:25-33: the docstring says CI does `pip install -e <stack>/sushicore` and the fallback list probes `repo.parent / "sushistack" / "sushicore"`, a path that no longer exists. D:/Projects/sushiruntime/.github/workflows/ci.yml:96-109 has the same SushiStack checkout and editable install, so F2 is a two-repository defect, and sushiblas's ci.yml:5 names that file as its model.

Recommendation: Fix F2 in sushiblas and sushiruntime in one change, and remove the dead candidate and the stale docstring sentence from conftest.py in the same commit.

Review: found by review.

#### L16. [low] Two version sources for one repository, kept in step by hand

Evidence: CMakeLists.txt:15 (`VERSION 0.1.0`) and cli/pyproject.toml:7 (`version = "0.1.0"`) are independent literals; nothing checks one against the other. include/SushiBLAS/version.hpp:42-49 is hand-written rather than generated (CMakeLists.txt:66 says so) though a configure-time check at CMakeLists.txt:72-76 guards it. versioning-and-release: 'version.hpp is generated from it' and 'every other copy is generated or checked against it'.

Recommendation: Decide whether the CLI package versions with the library; if yes, add a check tying pyproject to the CMake version. Keep or replace the hand-written version.hpp with a generated one, consistently with sushiruntime.

Review: confirmed. CMakeLists.txt:15 and cli/pyproject.toml:7 are independent literals. The version.hpp half is weaker than stated: CMakeLists.txt:72-80 fails configure on a mismatch, which meets 'generated or checked against it'. Only the pyproject half is a gap.

#### L17. [low] Committed work exists only on this machine

Evidence: `git log origin/main..main` shows aa5633e 'feat(cli): provision this checkout with sb setup, without hub' (the 2026-10-04 entry at docs/CHANGELOG.md:25); origin/main is at a commit dated 2026-09-22. Branch feature/sushicore_help_screen has no remote counterpart but is 0 commits ahead of main, so nothing unique sits on it.

Recommendation: Push main when the owner agrees. Delete the merged local branches feature/sushicore_help_screen and feature/blas-spatial-domain after confirmation.

Review: confirmed. origin/main is aee576a (2026-09-22), main is aa5633e, one ahead. Both feature branches are 0 ahead of main; feature/blas-spatial-domain tracks origin, feature/sushicore_help_screen has no remote.

#### L18. [low] .gitignore gaps and an ignored 206 MB build tree with stale install metadata

Evidence: .gitignore has no rule for .pytest_cache/, venv/, .venv/, build-asan/, .idea/, *.log or dist/; cli/.pytest_cache/ is ignored only by the .gitignore pytest writes inside it (`git check-ignore -v cli/.pytest_cache` matches no repository rule). .gitignore:34 `cli/*.egg-info/` covers only cli/. On disk: build/ is 206 MB; cli/sushiblas_cli.egg-info/PKG-INFO line 7 says `Requires-Dist: sushicore>=0.1.0` against cli/pyproject.toml:12 `sushicore>=0.7.0`.

Recommendation: Add .pytest_cache/, venv/, .venv/, build-*/, *.egg-info/, dist/ to .gitignore. Reinstall the CLI (`pip install -e cli`) so the editable metadata matches pyproject; that is the owner's step to run.

Review: confirmed. .gitignore has 35 lines and none of the named rules; git check-ignore -v matched no repository rule for cli/.pytest_cache; PKG-INFO:7 says sushicore>=0.1.0; build/ is 206M.

#### L19. [low] Repository URL and product name are cased differently across files

Evidence: `git remote -v`: https://github.com/SushiSystems/SushiBLAS.git. sushi-module.toml:12: `repo = "https://github.com/sushisystems/sushiblas.git"`. Local folder and module name are `sushiblas`, CMake project is `SushiBLAS`, the Python distribution is `sushiblas-cli`. GitHub resolves both URLs, so nothing breaks today.

Recommendation: Pick one spelling of the remote URL for every manifest across the stack and use it in sushi-module.toml and the remote alike.

Review: confirmed. git remote -v gives SushiSystems/SushiBLAS.git, sushi-module.toml:12 gives sushisystems/sushiblas.git. Low severity is right.

### Not checked

- Contents of build/ beyond its top level and size (out of scope by instruction).
- Whether the remote SushiSystems/SushiStack main still contains a sushicore/ directory; F2 rests on the local checkout and the changelog only.
- Whether the CI workflow has ever run or is currently green on GitHub; no network or gh call was made.
- Git history for files that were once tracked and later removed (large blobs in old commits); only the current tree and .git size (2.9 MB) were looked at.
- A full count of changelog entries that break the one-line rule; only the 240-character scan (0 physical lines over) and one sample at lines 121-125 were examined.
- Tier and layering declarations inside cmake/ and whether src/ includes respect them (belongs to check_layering.py, which does not exist here).
- File and folder naming (snake_case, one symbol per header) across include/, src/, kernels/ and tests/.
- Contents of cli/config.toml, cli/sushistack.deps.toml, CMakePresets.json, NOTICE.md and the two docs/agent/reports files beyond their existence.
- tests/package/ and tests/common/ against the tests/{unit,integration,regression,common} shape beyond the directory listing; tests/functional/ adds a level the skill does not name and was not assessed as a finding.
- Licence headers in source files and the licence choice itself (a separate dimension).
- The remote-only branch state of origin/feature/blas-spatial-domain beyond the local comparison.

## Code shape

### Facts

- 361 tracked files: 220 .cpp, 84 .hpp, 1 .h, 16 .py, 14 .md, 9 .cmake. Languages are C++17 (SYCL) and Python; there is no TypeScript and no GLSL.
- The repository is a single static library, not a modules/<tier>/<module> tree. Its de facto modules are folders: core, tensor/storage, engine facade and dispatch seam, operation sources, kernels, io, experimental, cli, tests, examples.
- include/SushiBLAS: 44 files, 11,909 lines (public headers: core 5, engine 27, public kernels 4, experimental 2, root 6).
- src: 115 files, 8,036 lines. src/engine holds 112 one-operation .cpp files (blas 19, logic 13, elementwise 32, nonlinear 8, random 16, reductions 16, spatial 8); engine.cpp and io.cpp sit at src/ root.
- kernels (private headers): 38 files, 4,660 lines (reference 16, sycl 18, 4 umbrella headers).
- cli/sushiblas: 11 Python files, 807 lines; cli/tests: 6 files, 348 lines. The CLI is a thin Typer layer over the external sushicore package.
- tests: 109 files, about 11,900 lines. tests/functional/unit 98, integration 2, regression 2, tests/common 4, tests/package 2. examples: 3 files, 160 lines. cmake: 10 files, 384 lines.
- Declared dependency direction (docs/ARCHITECTURE.md section 2): Engine -> per-domain op views -> dispatch helpers -> kernels (reference, sycl) -> Tensor/Storage -> SushiRuntime. core/ (common, format, logger, traits) sits under everything.
- Observed direction: src/*.cpp -> include/SushiBLAS/engine/** and kernels/**; kernels/** -> include/SushiBLAS/core and also up into include/SushiBLAS/engine (launch_1d, spatial/geometry, reduce_1d, device_policy); engine.hpp includes every op header while every dispatch header includes engine.hpp.
- External dependencies visible in public headers: <sycl/sycl.hpp> in 30 of 44 public headers, and SushiRuntime (RuntimeContext, TaskGraph, sushi_ptr, ResourceRegion, IAllocatorStrategy, span).
- Twenty largest source files (lines): tests/functional/unit/test_gemm.cpp 995; include/SushiBLAS/engine/reduce_1d.hpp 868; include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp 825; include/SushiBLAS/engine/math/reductions/dispatch.hpp 767; tests/functional/unit/test_axis_reductions.cpp 695.
- Largest files, continued: include/SushiBLAS/engine/launch_1d.hpp 581; tests/functional/unit/test_convert.cpp 572; include/SushiBLAS/engine/blas/dispatch.hpp 529; include/SushiBLAS/tensor.hpp 500; include/SushiBLAS/core/format.hpp 472.
- Largest files, continued: tests/functional/regression/test_regression.cpp 462; include/SushiBLAS/core/logger.hpp 461; include/SushiBLAS/engine/device_policy.hpp 417; tests/functional/unit/test_dispatch_seam.cpp 415; cli/sushiblas/services/project.py 406.
- Largest files, continued: include/SushiBLAS/engine/dispatch.hpp 402; include/SushiBLAS/engine/math/elementwise.hpp 398; include/SushiBLAS/engine/spatial.hpp 397; include/SushiBLAS/engine/reduce_axis.hpp 396; tests/functional/unit/test_storage_offset.cpp 390.
- Test layout: one GoogleTest binary, sources under tests/functional/{unit,integration,regression}, CTest labels unit/integration/regression set in tests/CMakeLists.txt:188-198. There is no tests/golden and no tests/benchmark. tests/package is an out-of-tree find_package consumer. cli/tests holds 5 pytest files.
- No test calls any IO method: save, load, save_npy, load_npy, save_bin, load_bin, to_string and print have zero references under tests/. core/format.hpp (472 lines, hand-written formatter) and core/logger.hpp have no direct test.
- No dedicated test file, but exercised inside a shared one (test_distributions, test_initializers, test_reductions, test_axis_reductions): bernoulli, discrete_uniform, exponential, log_normal, poisson, he_*, xavier_*, set_seed, argmax, argmin, logsumexp, prod, std, var, sum, max_along, sum_along, pool_max_backward, upsample_nearest.
- CLI services without a test: cli/sushiblas/services/diag.py, services/discovery.py, env.py, and the build/test/run/clean/doxygen functions in services/project.py (only its suite table is tested, in cli/tests/test_suites.py).
- Source-comment census over 305 C++ files: 290 have no @file block, 305 have no @author, 72 have no license block (69 under tests/functional/unit, 2 integration, 1 tests/common), 0 use @date, every header has #pragma once.
- Comment census, continued: 32 TODO lines in 23 files; 532 runs of stacked // lines in 120 files; 261 of 747 Doxygen blocks are longer than eight lines (77 files); 31 separator comment lines in 5 test files; 35 comment lines narrate history (used to, previously, now lives).
- Log census outside logger.hpp: 43 SB_LOG_INFO, 3 SB_LOG_DEBUG, 1 SB_LOG_WARN, 0 SB_LOG_ERROR direct calls; 213 SB_THROW_IF sites (21 written as SB_THROW_IF(true, ...)). The library has no catch block at all; the only one is examples/io_example.cpp:71.
- The repository carries no tools/ folder (no check_source_comments.py) and no .clang-format.
- Module verdict, core (include/SushiBLAS/core): needs targeted fixes. format.hpp and common.hpp read cleanly; logger.hpp has a lost-message path, no categories and two undocumented classes (F3, F4).
- Module verdict, tensor/storage (tensor.hpp, storage.hpp): sound in shape; needs the comment pass only. storage.hpp was read in full, tensor.hpp only in outline.
- Module verdict, engine facade and dispatch seam (include/SushiBLAS/engine.hpp, engine/**): needs restructuring. Seven hand-written dtype switches, three recorder shapes, and an execution mode five domains ignore (F1, F2, F7, F8).
- Module verdict, operation sources (src/engine/**): needs targeted fixes. The elementwise, nonlinear and most logic files are uniform one-call bricks; random, reductions and BLAS level 1/2 carry duplicated boilerplate and embedded kernels (F1, F6, F9, F10).
- Module verdict, kernels (kernels/**, include/SushiBLAS/kernels/**): needs targeted fixes. Reference and SYCL siblings match each other; the upward includes, the level1 single-file shape and SYRK reaching into GEMM's Detail are the gaps (F11, F12).
- Module verdict, io (include/SushiBLAS/io.hpp, src/io.cpp): needs restructuring. One class owns three file formats, text rendering and console output, with unchecked reads and no tests (F5).
- Module verdict, experimental and sparse stubs: dead code, to be removed or moved to the backlog (F13).
- Module verdict, cli (cli/sushiblas): needs targeted fixes. Delegation to sushicore is clean; project.py mixes five commands and writes to the terminal from the service layer (F16).
- Module verdict, tests: needs targeted fixes. Coverage of operations is broad; layout, file headers, tolerance and seeding rules are not met (F17, F18).

### Corrections from review

- Public headers including <sycl/sycl.hpp>: 27 of 44 (28 mention sycl::), not 30 of 44.
- Exact-float assertions under tests/: 64 (EXPECT/ASSERT_FLOAT_EQ and DOUBLE_EQ) against 25 NEAR, not 73 against 16. The sum, 89, is the same, so nine NEAR assertions were counted as exact.
- Direct add_task sites outside the shared recorder: 27, not 22. src/engine/math/random/constant.cpp alone has 7; the dispatch headers have 15 and the other five source files one each.
- 'catch sites: examples/io_example.cpp:71 only' holds for include/, src/, kernels/ and examples/. tests/ has 8 more catch sites (test_convert.cpp:155, test_count_nonfinite.cpp:247, test_dispatch_seam.cpp:408, test_spatial_channels.cpp:180 and :194, test_spatial_im2col.cpp:268, test_spatial_upsample.cpp:184 and :198).
- F11 says 'These four headers' and lists five (launch_1d, reduce_1d, reduce_axis, device_policy, spatial/geometry).
- F1 title says 'at least nine' copies; its own evidence list plus src/engine/math/random/orthogonal.cpp:108 (not listed) gives eleven switch sites and three if-chains in io.cpp.
- tests/: 109 tracked files hold 12,126 lines including tests/CMakeLists.txt, against the stated 'about 11,900'; the per-folder figures in verification_output (568 + 10,996 + 350) exclude the CMake file.
- Log census by my grep: 44 SB_LOG_INFO and 214 SB_THROW_IF outside logger.hpp, one more than stated in each case; 21 SB_THROW_IF(true matches.

### Findings

#### C1. [high] The runtime-dtype switch is written out at least nine times, so a new DataType edits nine places

Evidence: Shared helper exists at include/SushiBLAS/engine/dispatch.hpp:334 (with_element_type, switch at :341). Hand-written copies: include/SushiBLAS/engine/task.hpp:115-140; include/SushiBLAS/engine/math/reductions/dispatch.hpp:158-195 (three arms that differ only in the cast type); include/SushiBLAS/engine/math/random/dispatch.hpp:~196-218; src/engine/math/random/constant.cpp:51-135 (seven arms, each a full add_task with the same body); src/engine/math/random/shuffle.cpp:100-116; src/engine/math/elementwise/convert.cpp:154 and :258; src/io.cpp:139-149, :250-269, :280-286 (if-chains); plus include/SushiBLAS/core/common.hpp:88 and :158. task.hpp:103-114 says the copy is 'kept in step by hand' because of a compiler crash.

Recommendation: Make one dtype-to-type table the single brick (a typed visitor in engine/dispatch.hpp) and route constant, shuffle, execute_reduction, execute_random and io through it. Where the clang crash forces a spelled-out switch, keep exactly one such fallback. The module wave should start at src/engine/math/random/constant.cpp and reductions/dispatch.hpp:158, which are plain duplication with no compiler excuse recorded.

Review: confirmed. Opened engine/dispatch.hpp:334-339, task.hpp:103-140, reductions/dispatch.hpp:158-195, random/dispatch.hpp:196-218, shuffle.cpp:100-116, io.cpp:139-149/250-269/280-286; grep also finds constant.cpp:51, convert.cpp:152 and :256 (audit says 154/258) and one the audit did not list, src/engine/math/random/orthogonal.cpp:108.

#### C2. [high] Engine::set_mode is honoured only by BLAS and spatial; five domains ignore it without saying so

Evidence: get_mode() is read in src/engine/blas/level1 (9 files), level2 (7), level3 (2), src/engine/spatial (8) and include/SushiBLAS/engine/blas/dispatch.hpp:499. No file under src/engine/math, src/engine/logic, or include/SushiBLAS/engine/{math,logic} reads it; ExecutionMode appears only in include/SushiBLAS/engine/task.hpp:169 and :178. docs/ARCHITECTURE.md:158-160 states the non-BLAS domains 'follow the same pattern inline'. The enumerator comments in include/SushiBLAS/engine.hpp:51-54 are also stale ('Always use SYCL/MKL', 'Run both and compare'), and ARCHITECTURE.md:116 itself says VERIFY compares nothing.

Recommendation: Decide (owner decision, public behaviour) whether the mode is a library-wide contract. If it is, elementwise, nonlinear, logic, reductions and random need a reference path through dispatch_with_mode; if it is not, FORCE_REFERENCE and VERIFY must throw or warn in those domains. Fix the enumerator comments in engine.hpp:51-54 either way.

Review: confirmed. get_mode() users recomputed: level1 9, level2 7, level3 2, spatial 8, blas/dispatch.hpp 1, engine.hpp 1; zero hits for get_mode or ExecutionMode under src/engine/{math,logic} and include/.../{math,logic}. engine.hpp:51-54 and ARCHITECTURE.md:116, :158-160 read as quoted.

#### C3. [high] SB_ASSERT loses its message, and the async logger can hang on shutdown

Evidence: include/SushiBLAS/core/logger.hpp:438-444: SB_ASSERT calls SB_LOG_ERROR and then std::abort(). In the default async mode (logger.hpp:228-232) the message is only queued, so abort() ends the process before the worker thread writes it. logger.hpp:171-175: the destructor stores running_ = false without holding mutex_ and then notifies; the worker at :137-138 evaluates the predicate under the lock, so the store can land between the predicate check and the wait and the notify is lost, leaving worker_.join() at :178 blocked. <cstdlib> is not included for std::abort.

Recommendation: Give the logger a synchronous flush used by SB_ASSERT (and any future FATAL), and set running_ under mutex_ before notifying. Add a unit test for both paths; logger.hpp has none today.

Review: confirmed. logger.hpp read: SB_ASSERT logs then std::abort() while dispatch_log defaults to the async queue; the destructor stores running_ without mutex_ before notify_one, the worker predicate runs under the lock, so the lost wakeup is possible. <cstdlib> is absent from the include list at :26-38. Derived from reading, not reproduced.

#### C4. [high] IO is one class with four responsibilities, unchecked reads, a sibling that mishandles device memory, and no tests

Evidence: src/io.cpp holds the .sushi format (:45-102), raw binary (:104-123), .npy (:125-235), text rendering (:237-295, :302-364) and console printing (:297-300). load() checks magic, rank and shape only (:85-89); header.version, dtype and layout are never compared although include/SushiBLAS/io.hpp:62 promises metadata verification. No read is checked for short input (:83, :97, :101, :122, :228, :232). load_npy verifies shape with a substring search and skips the check when 'shape' is absent (:215-219), and never checks dtype or fortran_order (TODO at :213-214). save_bin/load_bin (:104-123) dereference t.data() on the host with no Device-strategy branch, while save, load, save_npy and load_npy each repeat that branch (:65-72, :94-101, :182-189, :225-232). All paths write num_elements bytes from data(), which is wrong for a strided view (docs/ARCHITECTURE.md:173-178 says element_span differs). SushiHeader (:35-43) is a raw struct dump with padding, declared in namespace SushiBLAS inside a .cpp. Zero tests reference any IO method.

Recommendation: Split into one reader/writer brick per format behind a shared 'copy tensor bytes to or from host' helper, a separate text renderer, and a status-returning load that validates every header field and every read count. Load the file-formats skill for the .sushi header. Add unit tests before touching behaviour. This is a boundary change and needs the owner's approval.

Review: confirmed. src/io.cpp:35-300 read in full: every cited line holds. load() checks magic, rank and shape only; no read is checked; save_bin/load_bin have no Device branch; load_npy uses a substring search guarded by shape_pos != npos. IO appears only in examples/io_example.cpp, never under tests/.

#### C5. [high] truncated_normal ignores its bounds and returns a plain normal draw

Evidence: src/engine/math/random/truncated_normal.cpp:36-40: '(void)a; (void)b;', an INFO line saying 'Falling back to normal', a TODO, then 'return normal(t, mean, stddev);'. INFO is compiled out of release builds (logger.hpp:82-88), so the fallback is invisible there. The only test, tests/functional/unit/test_distributions.cpp:84-89, calls it and asserts nothing.

Recommendation: Either implement the truncation or make the method throw until it exists; a fallback that changes the result must be WARN at minimum. Replace the test with one that asserts every sample lies in [a, b].

Review: confirmed. truncated_normal.cpp:36-40 and test_distributions.cpp:83-89 read; the test has no assertion. logger.hpp:83-88 sets the release level to WARN, so the INFO fallback line is compiled out.

#### C6. [medium] Logging breaks the logging skill in shape: no categories, no TRACE or FATAL, INFO on every operation launch, stdout as sink

Evidence: include/SushiBLAS/core/logger.hpp:97-104 defines OFF/ERR/WARN/INFO/DEBUG only; SB_LOG_* macros at :407-420 take no category and no <module>_log_category.hpp exists. Non-error lines go to stdout (logger.hpp:146, :200). 43 INFO calls fire once per recorded operation at execution, for example include/SushiBLAS/engine/math/elementwise/dispatch.hpp:93, :161, :303; include/SushiBLAS/engine/logic/dispatch.hpp:90; kernels/sycl/blas/level2/gemv.hpp:66; src/engine/math/random/constant.cpp:58-129. logger.hpp:74-80 records that this cost forced the release default down to WARN. Format specs are used in log calls (constant.cpp:58, src/engine/math/random/orthogonal.cpp:106, truncated_normal.cpp:38). src/io.cpp:299 writes to std::cout from library code.

Recommendation: Add a category argument and one K_<MODULE>_LOG_CATEGORY per domain, add TRACE and FATAL, move per-launch lines to TRACE or a counter, and send log output to stderr. Whether the stack-wide formatter keeps format specs is an owner decision, because include/SushiBLAS/core/format.hpp:60-64 implements them on purpose and SushiAI uses them.

Review: confirmed. LogType has OFF/ERR/WARN/INFO/DEBUG only, macros take no category, non-error lines go to stdout in both loggers, io.cpp:299 writes std::cout, truncated_normal.cpp:38 uses {:.4f}. My INFO count is 44, the audit's 43; the difference does not matter.

#### C7. [medium] Three recorder shapes for one job: 22 direct add_task sites rebuild metadata and access sets by hand

Evidence: The shared recorder is Internal::execute_typed_task (include/SushiBLAS/engine/task.hpp:92-100). Beside it, direct engine.get_graph().add_task calls: include/SushiBLAS/engine/logic/dispatch.hpp:87, :147, :208; engine/math/elementwise/dispatch.hpp:90, :158, :300; engine/math/nonlinear/dispatch.hpp:85, :149; engine/math/random/dispatch.hpp:191; engine/math/reductions/dispatch.hpp:161, :172, :183, :258, :371, :665; and in sources src/engine/logic/all.cpp, any.cpp, src/engine/math/elementwise/convert.cpp, src/engine/math/random/constant.cpp, orthogonal.cpp, shuffle.cpp. 'meta.task_type = ' is assigned at 51 sites in 38 files and the 'if (ptr) reads.push_back(t.region())' idiom appears 96 times (for example src/engine/blas/level1/axpy.cpp:45-58).

Recommendation: Introduce one task-record builder (name, op id, params, reads, writes from tensors) used by every domain, so an operation file states only its operands and its kernel. Level 1 and level 2 BLAS sources are the first candidates.

Review: confirmed. Defect is real; the count is low. add_task sites outside task.hpp, blas/utils.hpp and engine.hpp number 27 (15 in dispatch headers, 12 in sources because constant.cpp has 7), not 22. meta.task_type sites: 51, as stated.

#### C8. [medium] Engine and the op views depend on each other, and Engine is edited for every new domain

Evidence: include/SushiBLAS/engine.hpp:27-40 includes io.hpp and every op-family header and exposes blas(), io(), logic(), random(), nonlinear(), reductions(), spatial(), elementwise() at :98-140. Each view holds Engine& and every dispatch header includes <SushiBLAS/engine.hpp> back (task.hpp, blas/utils.hpp, blas/dispatch.hpp, logic/dispatch.hpp, math/*/dispatch.hpp). Engine also owns RNG seed state (:210-231, :238-239) next to graph ownership and tensor allocation (src/engine.cpp:40-55), which is three responsibilities. include/SushiBLAS/engine/blas.hpp:42 composes BLASOps by multiple inheritance from Level1, Level2, Level3 and SparseBLAS, each holding its own Engine&.

Recommendation: Have the views depend on a narrow interface (graph, context, mode, RNG stream) declared below both sides, and move the RNG stream into its own brick. Public interface shape, so ask before changing.

Review: confirmed. engine.hpp:27-40 includes io.hpp and every op family; blas/dispatch.hpp:53 includes engine.hpp back; seed_ and rng_offset_ sit beside graph_ at engine.hpp:234-239; create_tensor in src/engine.cpp:41-55; blas.hpp:42 inherits from four bases.

#### C9. [medium] Sibling reductions are shaped differently, and kernels live in three different places

Evidence: BLAS kernels sit under kernels/{reference,sycl}/; elementwise, nonlinear and logic kernels are lambdas at the call site (src/engine/math/elementwise/add.cpp:33-34); several reductions embed a full SYCL kernel in the operation source: src/engine/math/reductions/var.cpp:35-57, std.cpp, argmax.cpp:62, argmin.cpp:41, logsumexp.cpp:35, mean.cpp (single_task in 8 source files), while sum, asum, nrm2 and prod go through Reduce::reduce in include/SushiBLAS/engine/reduce_1d.hpp:771. LogicOps::all and any (src/engine/logic/all.cpp:40-62) are whole-tensor reductions that live in the logic module and record their own task instead of using Dispatch::execute_reduction. kernels/math.hpp:26-28 shows a reductions kernel folder was planned and never made.

Recommendation: Pick one home for a reduction kernel and move var, std, logsumexp, argmax, argmin, all and any to it; var and std fit the existing accumulator-policy shape in reduce_1d.hpp. The module wave should read every file in src/engine/math/reductions.

Review: confirmed. single_task is in 8 src files, of which 6 are reductions (argmax, argmin, logsumexp, mean, std, var) and 2 are random (orthogonal, shuffle). all.cpp:38-62 carries its own sycl::reduction kernel in the logic folder. kernels/math.hpp holds only commented-out includes.

#### C10. [medium] 32 TODO comments in source, including 13 identical 'batch support' lines and three on public declarations

Evidence: src/engine/blas/level1/axpy.cpp:43, copy.cpp:51, dot.cpp:47, rot.cpp:43, scal.cpp:40, swap.cpp:43; src/engine/blas/level2/gemv.cpp:46, ger.cpp:56, symv.cpp:57, syr.cpp:52, syr2.cpp:56, trmv.cpp:53, trsv.cpp:60; include/SushiBLAS/engine/blas/utils.hpp:63; include/SushiBLAS/io.hpp:56, :75, :92 (placed between the Doxygen block and the declaration); src/io.cpp:213, :214, :221; src/engine/math/reductions/argmax.cpp:62, argmin.cpp:41, logsumexp.cpp:35, std.cpp:34, var.cpp:34; src/engine/math/random/truncated_normal.cpp:39; include/SushiBLAS/engine/blas/sparse.hpp:45-58.

Recommendation: Move each item to docs/design/REMAINING_WORK.md as one backlog entry per topic and delete the comments.

Review: confirmed. Recomputed: 32 TODO/FIXME/HACK/XXX lines in 23 files. sparse.hpp:45-58, io.cpp:213-214 and :221, truncated_normal.cpp:39 opened.

#### C11. [medium] The private kernel layer includes upward into engine/ headers

Evidence: docs/ARCHITECTURE.md:95 says lower layers never depend up. kernels/ includes <SushiBLAS/engine/launch_1d.hpp> in 10 files (for example kernels/sycl/blas/level2/gemv.hpp:31), <SushiBLAS/engine/spatial/geometry.hpp> in 14, <SushiBLAS/engine/reduce_1d.hpp> in 1; include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp includes <SushiBLAS/engine/device_policy.hpp>. These four headers (launch_1d, reduce_1d, reduce_axis, device_policy, spatial/geometry) contain no Engine and are kernel-layer bricks filed under engine/.

Recommendation: Move the launch, reduce, device-policy and geometry headers to a layer below both engine/ and kernels/ (a folder decision for the owner), so the include graph matches the documented layering.

Review: confirmed. Includes of engine/launch_1d.hpp, spatial/geometry.hpp (14), reduce_1d.hpp (1) and device_policy.hpp (gemm.hpp:50) from the kernel folders exist; those four headers include only core/traits.hpp and each other, no Engine. The audit says 'four headers' and names five. The reverse edge also exists: engine/blas/dispatch.hpp:61-63 includes three kernel headers.

#### C12. [medium] Kernel siblings differ in file shape, and SYRK reaches into GEMM's Detail namespace

Evidence: Level 2 and level 3 have one kernel per file; level 1 packs every routine into kernels/reference/blas/level1/level1.hpp (208 lines) and kernels/sycl/blas/level1/level1.hpp (246 lines). kernels/sycl/blas/level3/syrk.hpp:119-120 calls Detail::strides_for and uses Detail::OperandStrides, which are declared inside include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp:71 and :104 as GEMM's internals. kernels/blas.hpp:37-59 mixes quoted and angle-bracket includes. The matrix-element accessor lambda is repeated in reference and SYCL GEMV (kernels/reference/blas/level2/gemv.hpp:44-47, kernels/sycl/blas/level2/gemv.hpp:73-76).

Recommendation: Split level1.hpp into one file per routine, and lift operand addressing (OperandStrides, strides_for, load_operand) into its own header that GEMM, SYRK and the level-2 kernels all include.

Review: confirmed. level1 is one file per backend, level2/3 one file per routine (ls). syrk.hpp:119-121 uses Detail::OperandStrides and strides_for declared in gemm.hpp:71 and :104. The A_at lambda is repeated at reference gemv.hpp:44-47 and sycl gemv.hpp:73-76.

#### C13. [medium] Dead code: unused export macro, empty kernel umbrella, declaration-only classes, an empty SparseBLAS base, a stray C++20 include

Evidence: include/SushiBLAS/core/export.hpp:26-35 defines SUSHIBLAS_API; nothing includes the header or uses the macro, and for this STATIC library it would expand to dllimport. kernels/math.hpp:26-28 contains only commented-out includes and no file includes it. include/SushiBLAS/experimental/lapack/linalg.hpp and experimental/signal/transforms.hpp declare methods with no definition anywhere (ARCHITECTURE.md:819-825 confirms). include/SushiBLAS/engine/blas/sparse.hpp:41-59 is a class with no methods that BLASOps still inherits (engine/blas.hpp:42). src/engine/math/nonlinear/gelu.cpp:24 includes <numbers>, a C++20 header, and uses nothing from it.

Recommendation: Delete export.hpp, kernels/math.hpp and the <numbers> include; remove SparseBLAS from BLASOps and the experimental headers from the installed set, recording the intent in the backlog. Deleting installed headers changes the public surface, so ask first.

Review: confirmed. SUSHIBLAS_API and core/export.hpp have no user outside the header itself; kernels/math.hpp is not included anywhere; no reference to the experimental headers from src, tests, examples or cmake; sparse.hpp:41-59 is a method-less class; gelu.cpp:24 includes <numbers> and never uses std::numbers.

#### C14. [medium] Source comments miss the Doxygen rules across the whole tree

Evidence: 290 of 305 C++ files have no @file block (only 15 do, for example include/SushiBLAS/engine/task.hpp:26) and none has @author. Blocks run far past the limits: task.hpp:26-47 is a 22-line file header; include/SushiBLAS/core/logger.hpp:340-367 is a 28-line symbol block; engine.hpp:190-209 is 20 lines; 261 of 747 blocks exceed eight lines. Design reasons and history live in source: include/SushiBLAS/core/traits.hpp:29-38, engine/blas/utils.hpp:42-45 and :105-110, core/common.hpp:44-60. Stacked // paragraphs: 532 runs in 120 files, for example src/engine/logic/all.cpp:39-45 and engine.hpp:161-165. Undocumented symbols: logger.hpp:124 (AsyncLogger), :192 (SyncLogger), :215, :221, :322; core/common.hpp:36-37 and :64-70 (enumerators); engine.hpp:234-239 (private members); src/io.cpp:35-43 fields. engine.hpp:51-54 documents enumerators with trailing // instead of /**< */. The repository has no tools/documentation/check_source_comments.py.

Recommendation: Run a comment pass per module: add the three-tag @file block, cut each block to @brief plus caller rules, and move the reasoning paragraphs into docs/design documents cited by path. Much of that prose is good design record and should be moved, not deleted. Add the checker so the pass stays done.

Review: confirmed. Recomputed over 305 C++ files: 290 without @file, 305 without @author. Separator lines: 31 in the five named test files. engine.hpp:51-54 trailing // enumerator comments and AsyncLogger/SyncLogger without blocks seen. The 532 stacked-run and 261/747 long-block counts were not recomputed.

#### C15. [medium] SYCL and SushiRuntime types are the public interface, and failures are exceptions, not statuses

Evidence: Every operation returns sycl::event (for example include/SushiBLAS/engine/spatial.hpp:99-391, engine/math/elementwise.hpp), public seams take sycl::queue& and std::vector<sycl::event> (include/SushiBLAS/engine/task.hpp:92-96), Tensor exposes sycl::device (tensor.hpp:260) and SushiRuntime::sushi_ptr<Storage>, and Storage exposes the allocator and strategy as public fields (storage.hpp:46-56). All 213 failure sites throw std::runtime_error through SB_THROW_IF (logger.hpp:452-459), including file and parse failures in src/io.cpp:58, :80, :85; cpp-code-style asks for a [[nodiscard]] status for failures outside the program and no exception across a module's public surface. Naming also departs from the style: enumerators are UPPER_SNAKE (core/common.hpp:36-70, engine.hpp:51-54), constants lack the K_ prefix (common.hpp:74, logger.hpp:114), ExecutionMode has no sized underlying type (engine.hpp:49), getters are noun-first with get_ and lack [[nodiscard]] (engine.hpp:86-92, :220).

Recommendation: These are public-interface decisions for the owner: whether SYCL in the surface is accepted for this library (the fused-kernel seam depends on it), and whether IO and allocation failures become status returns. Record the decision in a design document; rename enumerators and constants only together with an api-stability plan, since SushiAI consumes them.

Review: confirmed. storage.hpp:46-56 public allocator and strategy fields, tensor.hpp:69 public sushi_ptr<Storage>, task.hpp:92-96 sycl types in the seam, SB_THROW_IF throwing std::runtime_error at logger.hpp:452-459 and io.cpp:58/:80/:85, unsized UPPER_SNAKE ExecutionMode at engine.hpp:49-54 all read. Recommendation correctly defers to the owner.

#### C16. [medium] CLI: the service layer writes to the terminal, project.py holds five commands, and files miss the header rules

Evidence: cli/sushiblas/services/project.py (406 lines) contains build (:169), test (:217), test_package (:283), run (:372), clean (:393) and doxygen (:399) and calls console.header/info/error throughout (for example :171, :183-184, :389), which the python-code-style skill forbids for a service. Module-level singletons: _RUNNER and _DRIVER (project.py:78-79), _DIAG (services/diag.py:34), _INDEX (services/discovery.py:19), _ENV (env.py:23), PROFILE (config.py:29). project.py:194 calls shutil.rmtree(build_dir, ignore_errors=True), which hides a failed clean before a fresh configure. Constants are not K_-prefixed (project.py:60, :67; config.py:29), enum members are lower case (project.py:34-58). No .py file has a license header; module docstrings exceed six lines (config.py:1-16, project.py:1-13); functions at project.py:74, :169, :372, :393, :399 have no docstring. The package sits at cli/sushiblas rather than source/<package>.

Recommendation: Split project.py into one service per command that returns a result object, and let cli.py do the printing. Pass the runner and driver in instead of building them at import. Replace ignore_errors=True with a reported failure.

Review: confirmed. project.py is 406 lines with build :169, test :217, test_package :283, run :372, clean :393, doxygen :399 and console.* calls throughout; _RUNNER/_DRIVER at :78-79, _DIAG, _INDEX, _ENV, PROFILE found, plus _MODULE at config.py:43 which the audit omits. rmtree(ignore_errors=True) is at :193 and again at :310.

#### C17. [medium] Tests: layout, file headers and determinism rules are not met

Evidence: All C++ tests live under tests/functional/{unit,integration,regression}; the testing skill puts unit tests beside the module and integration, regression, golden and benchmark at the tests/ root. 72 test files open with #include and no license or @file block (for example tests/functional/unit/test_add.cpp). 73 assertions use EXPECT_FLOAT_EQ or EXPECT_DOUBLE_EQ against 16 EXPECT_NEAR, so the tolerance is not stated in the test. tests/common/test_utils.hpp:229 fills tensors with rand() and no srand or seed appears anywhere under tests/. Tests that need a device carry no device label; only two GTEST_SKIP sites exist (tests/functional/unit/test_gemm.cpp:902, :940). 31 separator comment lines sit in tests/functional/unit/test_axis_reductions.cpp, test_convert.cpp, test_storage_offset.cpp, test_gemm.cpp and test_count_nonfinite.cpp. tests/functional/regression/test_regression.cpp (462 lines) collects many defects in one file instead of one file named per defect.

Recommendation: Seed the shared fill helper with a value the test passes in, replace exact float comparisons with stated tolerances where the result is not bit-exact by contract, and split test_regression.cpp per defect. Moving the test tree is a layout decision for the owner.

Review: confirmed. Layout, rand() at test_utils.hpp:229 with no srand, two GTEST_SKIP sites and the separator census hold. The assertion split is wrong: 64 exact-float (EXPECT/ASSERT_FLOAT_EQ, DOUBLE_EQ) and 25 NEAR, not 73 and 16; the conclusion stands. For a one-module repository the yardstick layout is tests/{unit,integration,regression} at the root, so the defect is the extra functional/ level.

#### C18. [medium] No tests for IO, the formatter, the logger, or four CLI services

Evidence: Zero references under tests/ to save, load, save_npy, load_npy, save_bin, load_bin, to_string (src/io.cpp). No test includes core/format.hpp or calls Core::format directly, although include/SushiBLAS/core/format.hpp implements fill, align, width, precision and centring (:97-340). logger.hpp has no test for level filtering or shutdown. cli/sushiblas/services/diag.py, services/discovery.py, env.py and the command functions of services/project.py have no test file; cli/tests does not mirror the source paths.

Recommendation: Add test_format, test_logger and test_io (round-trip per dtype, per layout, strided view, truncated file, wrong dtype) before the IO restructuring in F5, and one pytest file per CLI source file.

Review: confirmed. No test calls engine.io(); the only callers are examples/io_example.cpp:46-67. core/format.hpp and logger.hpp are included only by tests/common/test_environment.hpp and test_main.cpp for setup. cli/tests holds test_config, test_console_lazy, test_help_screen, test_provision_commands, test_suites only.

#### C19. [medium] IO load paths write into a tensor without draining the recorded task graph

Evidence: D:/Projects/sushiblas/src/io.cpp: save (:55), save_bin (:106), save_npy (:127) and to_string (:239) call engine_.execute().wait() first. load (:77-102), load_bin (:115-123) and load_npy (:194-235) do not, and they are not recorded as tasks either. A tensor with a pending recorded write is filled from the file and then overwritten when the graph next executes; a pending reader sees the file's bytes or the old ones depending on call order. Derived from reading, not reproduced.

Recommendation: Decide once how IO relates to the graph and put it in the shared host-copy brick that F5 proposes: either every load drains the graph before the copy, or load is recorded as a task with the tensor's region as a write. Add a test that records an operation, loads, executes and checks the result.

Review: found by review.

#### C20. [medium] Nine random distributions reject complex dtypes inside the task, with a message that names no dtype

Evidence: D:/Projects/sushiblas/src/engine/math/random/bernoulli.cpp:41-45, and the same block at line 43-45 of discrete_uniform.cpp, exponential.cpp, log_normal.cpp, poisson.cpp, he_normal.cpp, he_uniform.cpp, xavier_normal.cpp, xavier_uniform.cpp: 'if constexpr (Core::is_complex_v<T>) SB_THROW_IF(true, "Unsupported data type for <op> operation.")' runs at graph execution. execute_random's record-time guard (include/SushiBLAS/engine/math/random/dispatch.hpp:166) rejects only integer and HALF. F20 names shuffle and execute_random only, so the module wave would not look here. A diff of bernoulli.cpp against exponential.cpp shows the nine files share the whole frame and differ in four lines.

Recommendation: Give execute_random a Domain parameter, as execute_typed_task has, so each distribution states REAL or REAL_AND_COMPLEX and the rejection happens at record time through Dispatch::require_domain. The in-lambda throw and the nine copies of it then go away.

Review: found by review.

#### C21. [medium] The include graph between engine/ and the kernels runs both ways, and kernels have two homes

Evidence: D:/Projects/sushiblas/include/SushiBLAS/engine/blas/dispatch.hpp:61-63 includes kernels/blas/level3/gemm_epilogue.hpp, kernels/sycl/blas/level3/gemm.hpp and kernels/reference/blas/level3/gemm.hpp, while include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp:50 includes engine/device_policy.hpp. GEMM and its epilogues live under include/SushiBLAS/kernels/ (public, installed); every other kernel lives under the private kernels/ folder. F11 reports only the upward edge and F12 does not mention the split.

Recommendation: Treat this together with F11: once launch, reduce, device-policy and geometry move below both layers, the remaining edge is engine -> kernels only. Record in a design document why GEMM is public (the fused-kernel seam) so the split between the two kernel folders is a stated rule, or move the rest to match. Folder and public-surface decisions belong to the owner.

Review: found by review.

#### C22. [medium] Audit quality, SOLID shape of recommendations: two are not brick-shaped as written

Evidence: F19 recommends owning the device profile 'on the Engine or the RuntimeContext'; F8 already counts three responsibilities on Engine (D:/Projects/sushiblas/include/SushiBLAS/engine.hpp:210-239), so the Engine option contradicts it. F4 bundles five changes (categories, two new levels, moving per-launch lines, stderr sink, format specs) in one recommendation with no order; the category header and the level set are the interface and must land before any call site moves. The other structural recommendations (F1 one dtype table, F5 one reader/writer per format over a shared host-copy helper, F7 one record builder, F9 one home for reduction kernels, F12 a shared operand-addressing header) are brick-shaped.

Recommendation: Drop the Engine option from F19 and name a separate device-profile brick passed to the launch helpers. Split F4 into an ordered pair: logger interface first (category type, TRACE and FATAL, stderr), call-site pass per domain second.

Review: found by review.

#### C23. [medium] Audit quality, humanizer register: plain, with three slips of fact inside the prose

Evidence: The findings state file, line and number and carry no filler connectives, evaluative adjectives or closing summaries. The slips are factual, not stylistic: 'These four headers' followed by five names (F11); 'at least nine times' against eleven sites in its own evidence (F1); '22 direct add_task sites' against 27 (F7). F17 presents the 73/16 assertion split as measured when the real split is 64/25.

Recommendation: Correct the four figures before the module wave uses them; no rewrite of the prose is needed.

Review: found by review.

#### C24. [low] Hidden process-wide state in public headers

Evidence: include/SushiBLAS/engine/device_policy.hpp:200-214 keeps a function-static, deliberately leaked device-profile cache and mutex. include/SushiBLAS/core/logger.hpp:215-232 keeps three function-static objects (the sync flag, SyncLogger, AsyncLogger), switched through the SB_LOGGER_SET_SYNC_MODE macro at :429. cpp-code-style asks for dependencies to arrive through the constructor, never through a global or singleton.

Recommendation: Own the device profile on the Engine or the RuntimeContext and pass it to the launch helpers; make the log sink an object the application installs. Both touch the public seam, so raise them with F8.

Review: confirmed. device_policy.hpp:200-214 holds a leaked static deque and static mutex; logger.hpp keeps three function statics. Low severity is right. The recommendation to own the profile on the Engine would add a fourth responsibility to the class F8 already calls overloaded; RuntimeContext or a separate profile brick is the consistent target.

#### C25. [low] Some dtype rejections happen at execution time, not at record time

Evidence: src/engine/math/random/shuffle.cpp:100-116 switches on dtype inside the task body and throws for INT32/INT64 from within the graph execution; include/SushiBLAS/engine/math/random/dispatch.hpp:~210-218 has the same in-task throw (a record-time guard exists at :166). docs/ARCHITECTURE.md:416-418 states out-of-domain dtypes are rejected at record time, 'where the caller can still act on it'.

Recommendation: Call Dispatch::require_domain at record time in shuffle and in every recorder that still switches inside the task; the consolidation in F1 removes the in-task throws.

Review: confirmed. shuffle.cpp:100-116 throws from the default arm inside the task for INT32/INT64 with no record-time guard; random/dispatch.hpp:196-218 keeps an in-task throw behind the record-time guard at :166. ARCHITECTURE.md:417-418 states record-time rejection.

### Not checked

- Line-by-line reading of include/SushiBLAS/tensor.hpp (outline only), engine/reduce_1d.hpp, engine/reduce_axis.hpp, engine/launch_1d.hpp, engine/device_policy.hpp, engine/blas/dispatch.hpp and core/format.hpp: only their declarations were listed, not their bodies.
- include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp and kernels/reference GEMM, SYRK, TRSM, and every spatial kernel under kernels/{reference,sycl}/spatial: not read beyond include lists and the SYRK Detail usage.
- kernels/sycl/random/philox.hpp and orthonormalize.hpp: not read.
- src/engine/spatial/*.cpp (8 files, 1,255 lines), src/engine/blas/level2 and level3 sources, src/engine/math/elementwise/convert.cpp (275 lines) and most of src/engine/math/random and reductions: sampled by grep only; axpy.cpp, var.cpp, gelu.cpp, add.cpp, sin.cpp, equal.cpp, all.cpp, constant.cpp, truncated_normal.cpp, engine.cpp and io.cpp were read.
- include/SushiBLAS/engine/spatial/window.hpp and geometry.hpp, engine/math/safe_math.hpp, version.hpp: not read.
- Test bodies: no test file was read in full; counts come from pattern searches, so the 73 exact-float assertions were not individually judged against a bit-exactness contract.
- examples/example.cpp, examples/io_example.cpp and tests/package/main.cpp: not read.
- cli/sushiblas/cli.py, config.py, console.py, diag.py, discovery.py and the test_package function in project.py: read as outline and docstrings only. cli/tests bodies not read.
- cmake/*.cmake, CMakeLists.txt, CMakePresets.json, .github/workflows/ci.yml, Doxyfile: out of this dimension, not reviewed.
- Formatting against a .clang-format (none exists), brace style, include-group order and column limit: not measured.
- Numerical correctness of any kernel, thread safety of the task-graph use, and the stated clang crash that justifies the hand-written switches: not verified; nothing was built or run.
- The lost-wakeup path in the logger destructor (F3) and the strided-view behaviour of IO (F5) are derived from reading the code, not reproduced.
- The 'no tests' list is by name search for a method call under tests/; a method exercised only through another API would be missed.
- The repository-layout, dependencies, file-formats and platform-portability skills were not loaded; layout remarks rest on the skills that were (sushiskills, source-comments, logging, testing, cpp-code-style, python-code-style).
- The build/ directory present on disk was skipped as instructed.
