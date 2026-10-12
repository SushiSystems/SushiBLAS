# SushiBLAS manual

The front door is the [root README](../README.md): what the library is, how to build it and how
to run its tests. This index reaches every other document.

SushiBLAS is below 1.0.0 and a MINOR release may break its public interface; read
[the stability section](../README.md#stability) before you depend on it.

## Architecture

| Document | Holds |
| --- | --- |
| [Architecture](architecture/README.md) | The layers, the execution modes, the tensor, storage and memory model, the engine, the recorder and the views, the dispatch seams and fusion hooks, the kernels and routines of each family, the SushiRuntime dependency, packaging and the exception types |

## Guides

| Document | Holds |
| --- | --- |
| [Integration](guides/INTEGRATION.md) | How another CMake project consumes SushiBLAS, what it must supply, and how it writes one adapter over the interface |
| [FAQ](guides/FAQ.md) | The questions a newcomer asks, each answered from this manual |
| [The CLI](../cli/README.md) | Every `sb` command and option; the CLI's own README |
| [Examples](../applications/examples/README.md) | The two example programs the build compiles |

## Beside the code

| Document | Holds |
| --- | --- |
| [Modules](../modules/README.md) | The tiers, what each module owns, and a link to each module's README |
| [Tests](../tests/README.md) | The test programs, their labels and how to run them |
| [Tools](../tools/README.md) | The checkers for documentation, layering, comments and versions |
| [Configuration](../.config/README.md) | The configuration files a flag points a tool at |

## Reference

| Document | Holds |
| --- | --- |
| [Changelog](reference/CHANGELOG.md) | What changed, one line per change |
| [Glossary](reference/GLOSSARY.md) | The words these documents use |
| [Known issues](reference/KNOWN_ISSUES.md) | Defects that are recorded and not yet fixed |
| [Security policy](../.github/SECURITY.md) | How to report a vulnerability and what is in scope |
| [Code of conduct](../.github/CODE_OF_CONDUCT.md) | The conduct expected in project spaces |
| [API reference main page](reference/MAINPAGE.md) | The page Doxygen opens the generated reference with |

## Design

| Document | Holds |
| --- | --- |
| [Design map](design/README.md) | Which design document covers which topic |
| [GPU validation](design/GPU_VALIDATION.md) | The claims that wait for a run on GPU hardware |
| [Module split](design/MODULE_SPLIT.md) | How the library is divided into modules and tiers |
| [Public seam](design/PUBLIC_SEAM.md) | Three properties of the public interface and the reason for each |
| [Remaining work](design/REMAINING_WORK.md) | The backlog |

## Contributing

| Document | Holds |
| --- | --- |
| [Contributing](CONTRIBUTING.md) | How a change is built, tested, documented and committed |
| [Documentation style guide](DOCUMENTATION_STYLE_GUIDE.md) | How prose is written here |

Agent work folders are under `agent/`. Finished material is under `archive/` and is not edited.
Doxygen output is written by `sb docs` to `build/docs/api-site/`, outside this folder.
