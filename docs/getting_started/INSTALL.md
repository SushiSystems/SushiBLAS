# Install SushiBLAS

SushiBLAS installs in two ways. Both give the same checkout and the same `sb` command, and neither needs the other: SushiHub is not required to build SushiBLAS, and a checkout made without it joins a SushiHub workspace later without any change.

## Standalone

You need Git, Python 3.11 or later and CMake. You clone nothing else: `sb setup` fetches SushiRuntime, SushiLog and SushiProfiler at the tags `cli/sushistack.deps.toml` pins, and `sb build` builds a package of each. A checkout of SushiRuntime beside this one, or the folder `SUSHIRUNTIME_DIR` names, is used instead when it exists.

```bash
git clone https://github.com/SushiSystems/SushiBLAS.git
cd SushiBLAS
pip install -e ./cli
sb setup
sb build
sb test
```

`pip install` also installs `sushicore` from PyPI, at the version `cli/pyproject.toml` pins. Machine paths go in the gitignored `cli/config.local.toml`; the store is `~/.sushisystems`, or the folder named by `SUSHISYSTEMS_HOME`.

## With SushiHub

```bash
hub add sushiblas            # clone it into the workspace
hub install-cli sushiblas    # put sb on the path
hub link sushiblas <path>    # adopt a checkout you already have, instead of hub add
```

`hub link` records where the checkout is and changes nothing in it. `hub install` provisions the toolchain and the libraries of every module in the workspace, and `hub doctor` and `hub status` show what is installed. After either route, `sb build` and `sb test` are the commands above.
