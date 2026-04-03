#!/usr/bin/env python3
"""
Read a dynesty checkpoint from the Sombrero JAM dynamic-nested run script and
regenerate the current posterior summaries/checkplots without resuming sampling.

Typical usage
-------------
python sombrero_jam_checkpoint_inspect.py \
    --run-script /path/to/sombrero_jam_dynamic_fixedml_checkplots_parallel_checkpointed.py

python sombrero_jam_checkpoint_inspect.py \
    --run-script /path/to/sombrero_jam_dynamic_fixedml_checkplots_parallel_checkpointed.py \
    --checkpoint /path/to/sombrero_jam_dynamic_fixedml_checkpoint.save \
    --prefix sombrero_jam_dynamic_fixedml_manualcheck

This script imports the main run script so it reuses the same MGE paths, JAM
configuration, fixed M/L, PSF, quality cuts, and plotting functions.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from types import ModuleType

import numpy as np
from dynesty import DynamicNestedSampler


def import_module_from_path(path: str) -> ModuleType:
    os.environ["JAM_INSPECT_ONLY"] = "1"
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run script not found: {path}")

    module_name = os.path.splitext(os.path.basename(path))[0] + "_imported_for_checkpoint"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from path: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_checkpoint_path(cfg: ModuleType, explicit_checkpoint: str | None) -> str:
    if explicit_checkpoint:
        checkpoint = os.path.abspath(os.path.expanduser(explicit_checkpoint))
    else:
        checkpoint = os.path.join(cfg.OUTPUT_PATH, cfg.CHECKPOINT_FILENAME)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    return checkpoint


def resolve_output_path(cfg: ModuleType, explicit_output_path: str | None) -> str:
    if explicit_output_path:
        output_path = os.path.abspath(os.path.expanduser(explicit_output_path))
    else:
        output_path = os.path.abspath(os.path.expanduser(cfg.OUTPUT_PATH))
    os.makedirs(output_path, exist_ok=True)
    return output_path


def summarize_to_console(bundle: dict, results) -> None:
    print("\nPosterior summary from checkpoint (weighted median +/- 68% interval):\n")
    for line in bundle["summary_lines"]:
        print(line)
    print("\nDerived quantities (maximum-likelihood sample in checkpoint):\n")
    print(f"inclination = {bundle['inc_best']:.2f} deg")
    print(f"sigma_z/sigma_R = {bundle['ratio_best']:.3f}")
    print(f"M_BH = {bundle['mbh_best']:.3e} Msun")
    print(f"reduced chi2 reported by JAM = {bundle['out_vrms'].chi2:.3f}")
    print(f"best-fit JAM kappa for velocity check plot = {bundle['out_vel'].kappa:.3f}")
    print(f"nsamples currently in checkpoint = {len(results.samples)}")
    print(f"logZ = {results.logz[-1]:.6f} +/- {results.logzerr[-1]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate current JAM/dynesty posterior plots and summaries from a checkpoint file."
    )
    parser.add_argument(
        "--run-script",
        required=True,
        help="Path to the main dynamic nested JAM script that created the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the dynesty checkpoint .save file. Defaults to OUTPUT_PATH/CHECKPOINT_FILENAME from the run script.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Directory where regenerated outputs should be written. Defaults to OUTPUT_PATH from the run script.",
    )
    parser.add_argument(
        "--prefix",
        default="sombrero_jam_dynamic_fixedml_manualcheck",
        help="Prefix for regenerated txt/npz/plot files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for equal-weight posterior resampling. Defaults to the run script SEED + 999.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only write summary/sample files and skip PNG plot generation.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Do not raise an error if the checkpoint exists but contains zero samples yet.",
    )
    args = parser.parse_args()

    cfg = import_module_from_path(args.run_script)
    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint)
    output_path = resolve_output_path(cfg, args.output_path)

    print(f"Imported run script: {os.path.abspath(os.path.expanduser(args.run_script))}")
    print(f"Checkpoint file: {checkpoint_path}")
    print(f"Output path: {output_path}")
    print(f"Output prefix: {args.prefix}")

    # Make the imported module write its summary paths consistently to the chosen output directory.
    cfg.OUTPUT_PATH = output_path

    print("\nBuilding JAM model and prior from the imported run script configuration...")
    model, prior_cfg = cfg.build_model_and_prior()

    print("Restoring dynesty checkpoint (without resuming sampling)...")
    sampler = DynamicNestedSampler.restore(checkpoint_path)
    results = sampler.results

    if len(results.samples) == 0:
        msg = (
            "The checkpoint was restored successfully, but it contains zero samples so there is "
            "nothing to plot yet."
        )
        if args.allow_empty:
            print(msg)
            return
        raise RuntimeError(msg)

    seed = args.seed if args.seed is not None else int(getattr(cfg, "SEED", 0)) + 999
    rng = np.random.default_rng(seed)

    print("Writing summary/sample files and regenerating plots from checkpoint state...")
    bundle = cfg.write_all_outputs(
        results,
        model,
        prior_cfg,
        output_path,
        rng,
        prefix=args.prefix,
        partial=True,
        save_plots=(not args.skip_plots),
    )

    summarize_to_console(bundle, results)

    print("\nDone. Files written to:")
    print(output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
