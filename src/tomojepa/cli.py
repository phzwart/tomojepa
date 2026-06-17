"""Unified ``tomojepa`` command-line entry point.

Thin dispatcher over the per-subsystem entry points. Each subcommand forwards
its remaining arguments verbatim to the underlying ``argparse`` ``main()`` (so
``tomojepa train-ssl --help`` shows that tool's real options), and the
patch-retrieval CLI is mounted as a Typer subgroup.

For multi-GPU training, launch the module directly under ``torchrun`` rather
than through this dispatcher, e.g.::

    torchrun --nproc_per_node=4 -m tomojepa.ssl.train --data_dir /data ...
    torchrun --nproc_per_node=4 -m tomojepa.vitup.train --teacher_ckpt ...
"""
import importlib
import sys

import typer

from . import __version__

# Pass every remaining token straight through to the wrapped argparse parser,
# including --help (so the wrapped tool prints its own, authoritative help).
_PASSTHROUGH = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="tomojepa: tomography SSL pre-training, ViT-Up upsampling, and patch retrieval.",
)


def _forward(target: str, prog: str, args):
    """Import ``module:function`` and run it with ``sys.argv`` = [prog, *args]."""
    mod_name, fn_name = target.split(":")
    main_fn = getattr(importlib.import_module(mod_name), fn_name)
    argv_bak = sys.argv
    sys.argv = [prog, *list(args)]
    try:
        return main_fn()
    finally:
        sys.argv = argv_bak


@app.callback(invoke_without_command=True)
def _root(
    version: bool = typer.Option(
        False, "--version", help="Show the tomojepa version and exit."
    ),
):
    if version:
        typer.echo(f"tomojepa {__version__}")
        raise typer.Exit()


@app.command("train-ssl", context_settings=_PASSTHROUGH,
             help="LeJEPA/DINOv3 self-supervised pre-training (pass --help for flags).")
def train_ssl(ctx: typer.Context):
    _forward("tomojepa.ssl.train:main", "tomojepa train-ssl", ctx.args)


@app.command("validate", context_settings=_PASSTHROUGH,
             help="Intrinsic validation of trained encoders (effective rank, invariance).")
def validate(ctx: typer.Context):
    _forward("tomojepa.ssl.validate:main", "tomojepa validate", ctx.args)


@app.command("train-vitup", context_settings=_PASSTHROUGH,
             help="ViT-Up multi-scale feature distillation training.")
def train_vitup(ctx: typer.Context):
    _forward("tomojepa.vitup.train:main", "tomojepa train-vitup", ctx.args)


@app.command("infer-vitup", context_settings=_PASSTHROUGH,
             help="ViT-Up inference / PCA visualization of upsampled features.")
def infer_vitup(ctx: typer.Context):
    _forward("tomojepa.vitup.infer:main", "tomojepa infer-vitup", ctx.args)


# --- visualization / analysis subgroup -----------------------------------
viz_app = typer.Typer(add_completion=False, no_args_is_help=True,
                      help="Visualization & analysis tools.")


@viz_app.command("cascade", context_settings=_PASSTHROUGH,
                 help="Cascading PCA-RGB component maps for a checkpoint.")
def viz_cascade(ctx: typer.Context):
    _forward("tomojepa.viz.cascade_rgb:main", "tomojepa viz cascade", ctx.args)


@viz_app.command("compare", context_settings=_PASSTHROUGH,
                 help="A/B comparison curves from two runs' metrics.json.")
def viz_compare(ctx: typer.Context):
    _forward("tomojepa.viz.build_comparison:main", "tomojepa viz compare", ctx.args)


@viz_app.command("shapes", context_settings=_PASSTHROUGH,
                 help="Segment objects per slice and measure shape stats (needs [analysis]).")
def viz_shapes(ctx: typer.Context):
    _forward("tomojepa.viz.analyze_shapes:main", "tomojepa viz shapes", ctx.args)


@viz_app.command("probe", context_settings=_PASSTHROUGH,
                 help="Linear-probe shape/material from encoder features (needs [analysis]).")
def viz_probe(ctx: typer.Context):
    _forward("tomojepa.viz.probe_shapes:main", "tomojepa viz probe", ctx.args)


@viz_app.command("build-token-db", context_settings=_PASSTHROUGH,
                 help="[legacy] Build a single-file .npz token DB (superseded by `patchdb build`).")
def viz_build_token_db(ctx: typer.Context):
    _forward("tomojepa.viz.build_token_db:main", "tomojepa viz build-token-db", ctx.args)


@viz_app.command("query-patches", context_settings=_PASSTHROUGH,
                 help="[legacy] Brute-force query a .npz token DB (superseded by `patchdb query`).")
def viz_query_patches(ctx: typer.Context):
    _forward("tomojepa.viz.query_patches:main", "tomojepa viz query-patches", ctx.args)


@viz_app.command("zscore", context_settings=_PASSTHROUGH,
                 help="Interactive Dash z-score explorer on ViT-Up dense features (needs [interactive]).")
def viz_zscore(ctx: typer.Context):
    _forward("tomojepa.viz.zscore_explorer:main", "tomojepa viz zscore", ctx.args)


app.add_typer(viz_app, name="viz")


# --- patch-retrieval engine (its own Typer app) --------------------------
from .patchdb.cli import app as _patchdb_app  # noqa: E402

app.add_typer(_patchdb_app, name="patchdb",
              help="FAISS + DuckDB patch-retrieval engine (build/add/query/serve/mcp).")


if __name__ == "__main__":
    app()
