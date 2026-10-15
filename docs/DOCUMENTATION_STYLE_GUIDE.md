# Documentation style guide

Prose in this repository follows the `humanizer` skill of SushiSkills, in the language it is
written in. Where a document goes, how a changelog entry reads and how an agent's work folder is
named follow the `documentation` skill. Neither is restated here.

## What this repository adds

- A symbol, path, CMake option or command is written in a code span and spelled as the code
  spells it: `Core::DataType`, `SB_SYCL_TARGETS`, `sb test --package`.
- A path is written from the repository root, such as `modules/engine/engine/include/SushiBLAS/engine/task.hpp`. A
  header named alone, such as `engine/task.hpp`, is its include path after `SushiBLAS/`.
- A code block in the manual is taken from a file the build compiles
  (`applications/examples/`, `tests/package/`, a test or a source), trimmed and never invented,
  and the file is named under the block. Where no compiled file shows the thing, the block says
  that no file compiles it and which headers it was checked against.
- A name in a code span exists in the tree. A name in `SushiBLAS::Internal` or a `Detail`
  namespace appears only on an architecture page.
- The manual says what is. What changed goes to the changelog, and a page carries no "previously",
  "no longer" or "new in".
- Prose in the manual uses no em dash. The entry format of `reference/CHANGELOG.md` keeps its
  dash, because `tools/documentation/check_changelog.py` requires it.
- A figure states what was measured, on which device and when. A claim that was derived and not
  measured says so; [the GPU validation design](design/GPU_VALIDATION.md) holds the claims that
  wait for hardware.
- A decision or work item cited by its short name (D1, D3, SB-6) is defined in
  [the glossary](reference/GLOSSARY.md).
- The CLI's manual is [`cli/README.md`](../cli/README.md), beside the code it describes.
- `docs/reference/MAINPAGE.md` is the page Doxygen opens the generated reference with. The Doxyfile,
  `.config/doxygen/Doxyfile`, reads it by that path.

## Words

English, British spelling in prose ("licence" the noun, "optimisation"). Identifiers keep the
spelling the code uses, such as `K_LICENSE_LINES` or the `LICENSE` file.

## Pages not yet through the humanizer pass

`design/GPU_VALIDATION.md` has not been through the pass.
