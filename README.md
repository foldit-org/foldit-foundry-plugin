# foldit-foundry-plugin

A Foldit plugin for running [RC Foundry](https://github.com/RosettaCommons)
machine-learning models — RoseTTAFold3, RFdiffusion3, and LigandMPNN — as an
in-game backend.

It brings three AI operations to the app:

| Button | Op | Model | What it does |
| --- | --- | --- | --- |
| Predict | `rf3_predict` | RoseTTAFold3 | Re-predict the structure of the focused entity (or the whole structure when nothing is focused) |
| Design | `rfd3_design` | RFdiffusion3 | Generate a binder against the focused protein or ligand |
| MPNN | `mpnn_design` | LigandMPNN | Redesign the focused chain's sequence, holding the selected residues fixed |

It also exposes a `sequence_design` query that returns scored sequence
candidates for a future UI panel.

## How it runs

This is a `kind = "python"` plugin. Foldit hosts it out-of-process through
`foldit-python-host`; the entry module is `foundry_plugin`. The heavy model code
(the `foundry` package) lives under `deps/foundry` and is imported by the thin
plugin wrapper.

Model weights are large and are not committed — they live under
`assets/weights/rc_foundry/` and are resolved at load time. On Apple hardware
the models run on the MPS (Metal) GPU backend.

## Setup

Python plugins use [pixi](https://pixi.sh) for their environment. Set the plugin
up (create the env, resolve weights) through the workspace xtask:

```bash
cargo xtask setup-plugins foundry
```

The plugin subclasses the `PluginInterface` base class from the Foldit plugin
SDK; see the Foldit workspace docs ("Python and Native Plugins") for how the
host discovers and loads a Python plugin.
