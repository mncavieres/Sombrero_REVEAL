#!/usr/bin/env python3
"""
Monitor the NIRSpec JAM dynesty checkpoint and regenerate live diagnostic plots.

This is the NIRSpec analogue of scripts/muse_jam/checkplot_muse.py and is
meant to run alongside nested_free_nirspec.py or its constant/logistic wrappers.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import logging
import shutil
import sys
import tempfile
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nested_free_nirspec as nirspec


base = nirspec.base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore the NIRSpec dynesty checkpoint and regenerate live JAM checkplots."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the dynesty checkpoint file. Defaults to Config.output_dir / checkpoint_filename.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the checkplots should be written. Defaults to <run output>/checkpoint_checkplots.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="Polling cadence in seconds when running in watch mode.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one checkpoint restore/checkplot pass and exit.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Do not error if the checkpoint exists but contains zero samples yet.",
    )
    return parser


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def resolve_paths(args: argparse.Namespace, run_cfg: nirspec.Config | None = None):
    run_cfg = run_cfg if run_cfg is not None else nirspec.make_free_config()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_cfg.output_dir / run_cfg.checkpoint_filename
    output_dir = Path(args.output_dir) if args.output_dir else run_cfg.output_dir / "checkpoint_checkplots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = replace(run_cfg, output_dir=output_dir)

    return run_cfg, plot_cfg, checkpoint_path


def restore_checkpoint_safely(checkpoint_path: Path, output_dir: Path):
    with tempfile.NamedTemporaryFile(
        prefix="dynesty_checkpoint_",
        suffix=".save",
        dir=output_dir,
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        shutil.copy2(checkpoint_path, tmp_path)
        sampler = DynamicNestedSampler.restore(str(tmp_path))
        return sampler.results
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def save_corner_checkplot(cfg, results, n_mge: int) -> None:
    outpath = cfg.output_dir / "checkpoint_cornerplot.png"
    labels = base.get_parameter_labels(cfg, n_mge)
    ndim = len(labels)

    try:
        fig, _ = dyplot.cornerplot(
            results,
            show_titles=True,
            labels=labels,
            title_kwargs={"x": 0.65},
            title_fmt=".3f",
        )
        fig.set_size_inches(max(10.0, 1.6 * ndim), max(10.0, 1.6 * ndim))
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return
    except Exception as exc:
        logging.warning("Dynesty cornerplot failed, falling back to nested_free summary plot: %s", exc)

    base.save_posterior_plot(cfg, results, n_mge)
    fallback_path = cfg.output_dir / "posterior_samples.png"
    if fallback_path.exists():
        fallback_path.replace(outpath)


def save_trace_checkplot(cfg, results, n_mge: int) -> None:
    outpath = cfg.output_dir / "checkpoint_traceplot.png"
    labels = base.get_parameter_labels(cfg, n_mge)
    ndim = len(labels)

    try:
        fig, _ = dyplot.traceplot(
            results,
            labels=labels,
            show_titles=True,
            title_fmt=".3f",
        )
        fig.set_size_inches(14.0, max(8.0, 1.5 * ndim))
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return
    except Exception as exc:
        logging.warning("Dynesty traceplot failed, using fallback point trace view: %s", exc)

    samples = np.asarray(results.samples, dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    if samples.ndim == 2 and samples.shape[0] > 0:
        sample_index = np.arange(samples.shape[0])
        dims = [0, min(1, samples.shape[1] - 1), samples.shape[1] - 1]
        for ax, idx in zip(axes, dims):
            ax.scatter(sample_index, samples[:, idx], s=10, alpha=0.55)
            ax.set_xlabel("Sample index")
            ax.set_ylabel(labels[idx])
            if idx == 0:
                ax.set_yscale("log")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No samples available yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Checkpoint trace diagnostics (fallback point view)", fontsize=13)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_run_checkplot(results, output_dir: Path) -> None:
    outpath = output_dir / "checkpoint_runplot.png"
    nsamp = len(np.asarray(results.samples))

    enough_for_dynesty_runplot = (
        nsamp >= 20
        and np.asarray(results.logl).size >= 20
        and np.asarray(results.logz).size >= 5
    )

    if enough_for_dynesty_runplot:
        try:
            fig, _ = dyplot.runplot(results)
            fig.savefig(outpath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return
        except Exception as exc:
            logging.warning("Dynesty runplot failed, using fallback point-only run diagnostics: %s", exc)

    samples = np.asarray(results.samples, dtype=float)
    logl = np.asarray(results.logl, dtype=float)
    logz = np.asarray(results.logz, dtype=float)
    logzerr = np.asarray(results.logzerr, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    if nsamp > 0:
        sample_index = np.arange(nsamp)

        axes[0].scatter(sample_index, logl, s=14, alpha=0.65)
        axes[0].set_xlabel("Sample index")
        axes[0].set_ylabel("logL")
        axes[0].set_title("Point-only logL progression")

        axes[1].scatter(np.arange(logz.size), logz, s=14, alpha=0.65)
        axes[1].set_xlabel("Checkpoint iteration")
        axes[1].set_ylabel("logZ")
        axes[1].set_title("Point-only evidence progression")

        if logz.size == logzerr.size and logz.size > 0:
            axes[2].scatter(np.arange(logzerr.size), logzerr, s=14, alpha=0.65)
            axes[2].set_xlabel("Checkpoint iteration")
            axes[2].set_ylabel("logZ err")
            axes[2].set_title("Point-only evidence uncertainty")

        if samples.ndim == 2 and samples.shape[1] > 0:
            axes[3].scatter(sample_index, samples[:, 0], s=14, alpha=0.65)
            axes[3].set_xlabel("Sample index")
            axes[3].set_ylabel(r"$M_{\rm BH}$")
            axes[3].set_yscale("log")
            axes[3].set_title("Point-only BH progression")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No samples available yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Checkpoint run diagnostics (fallback point view)", fontsize=13)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_status_summary(
    cfg,
    checkpoint_path: Path,
    results,
    best_params: np.ndarray,
    n_mge: int,
    vrms_out,
) -> None:
    outpath = cfg.output_dir / "checkpoint_status.txt"
    summary = base.summarize_best_params(cfg, best_params, n_mge)

    with outpath.open("w", encoding="utf-8") as handle:
        handle.write("NIRSpec JAM checkpoint status\n")
        handle.write("============================\n\n")
        handle.write(f"checkpoint = {checkpoint_path}\n")
        handle.write(f"nsamples = {len(results.samples)}\n")
        handle.write(f"logZ = {float(np.asarray(results.logz)[-1]):.6f}\n")
        handle.write(f"logZerr = {float(np.asarray(results.logzerr)[-1]):.6f}\n")
        handle.write(f"beta_prescription = {cfg.beta_prescription}\n")
        handle.write(f"current_best_bh_mass = {summary['best_bh_mass']:.6e}\n")
        handle.write(f"current_best_ml = {summary['best_ml']:.6f}\n")
        handle.write(f"current_best_jam_reduced_chi2 = {float(vrms_out.chi2):.6f}\n")

        if "best_beta" in summary:
            handle.write(f"current_best_beta = {summary['best_beta']:.6f}\n")
        if "best_beta_ra" in summary:
            handle.write(f"current_best_beta_ra = {summary['best_beta_ra']:.6f}\n")
            handle.write(f"current_best_beta_0 = {summary['best_beta_0']:.6f}\n")
            handle.write(f"current_best_beta_inf = {summary['best_beta_inf']:.6f}\n")
            handle.write(f"current_best_beta_alpha = {summary['best_beta_alpha']:.6f}\n")
        if "best_beta_array" in summary:
            beta_arr = np.asarray(summary["best_beta_array"], dtype=float)
            handle.write(
                "current_best_beta_array = "
                + ",".join(f"{value:.6f}" for value in beta_arr)
                + "\n"
            )


def write_checkpoint_checkplots(
    run_cfg,
    plot_cfg,
    checkpoint_path: Path,
    kin,
    mge,
    results,
) -> None:
    if len(results.samples) == 0:
        raise RuntimeError("Checkpoint restored but contains zero samples.")

    n_mge = len(mge.surf_lum)
    best_params = base.get_best_fit_parameters(results)

    save_corner_checkplot(plot_cfg, results, n_mge)
    save_trace_checkplot(plot_cfg, results, n_mge)
    save_run_checkplot(results, plot_cfg.output_dir)
    vrms_out, _, _ = base.compute_bestfit_vrms_model(plot_cfg, kin, mge, best_params)
    base.save_vrms_bestfit_plot(plot_cfg, kin, mge, best_params)
    write_status_summary(plot_cfg, checkpoint_path, results, best_params, n_mge, vrms_out)


def maybe_refresh_from_checkpoint(
    run_cfg,
    plot_cfg,
    checkpoint_path: Path,
    kin,
    mge,
    allow_empty: bool,
) -> bool:
    results = restore_checkpoint_safely(checkpoint_path, plot_cfg.output_dir)
    nsamples = len(results.samples)

    if nsamples == 0:
        msg = f"Checkpoint exists but currently contains zero samples: {checkpoint_path}"
        if allow_empty:
            logging.info(msg)
            return False
        raise RuntimeError(msg)

    write_checkpoint_checkplots(run_cfg, plot_cfg, checkpoint_path, kin, mge, results)
    logging.info(
        "Wrote checkpoint checkplots from %s with %d samples",
        checkpoint_path,
        nsamples,
    )
    return True


def run_once(
    run_cfg,
    plot_cfg,
    checkpoint_path: Path,
    kin,
    mge,
    allow_empty: bool,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    maybe_refresh_from_checkpoint(run_cfg, plot_cfg, checkpoint_path, kin, mge, allow_empty)


def run_watch_loop(
    run_cfg,
    plot_cfg,
    checkpoint_path: Path,
    kin,
    mge,
    poll_seconds: float,
    allow_empty: bool,
) -> None:
    last_mtime: float | None = None
    missing_reported = False

    logging.info("Watching checkpoint: %s", checkpoint_path)
    logging.info("Writing checkplots to: %s", plot_cfg.output_dir)

    while True:
        if not checkpoint_path.exists():
            if not missing_reported:
                logging.info("Checkpoint not found yet, waiting...")
                missing_reported = True
            time.sleep(poll_seconds)
            continue

        missing_reported = False

        try:
            mtime = checkpoint_path.stat().st_mtime
        except OSError:
            time.sleep(poll_seconds)
            continue

        if last_mtime is None or mtime > last_mtime:
            try:
                refreshed = maybe_refresh_from_checkpoint(
                    run_cfg,
                    plot_cfg,
                    checkpoint_path,
                    kin,
                    mge,
                    allow_empty,
                )
                if refreshed:
                    last_mtime = mtime
            except Exception as exc:
                logging.warning("Could not restore checkpoint yet: %s", exc)

        time.sleep(poll_seconds)


def run_cli(run_cfg: nirspec.Config | None = None) -> None:
    setup_logging()
    args = build_parser().parse_args()

    run_cfg, plot_cfg, checkpoint_path = resolve_paths(args, run_cfg=run_cfg)

    logging.info("Loading static NIRSpec inputs")
    kin = base.load_kinematics(run_cfg)
    mge = base.load_mge_inputs(run_cfg)

    if args.once:
        run_once(run_cfg, plot_cfg, checkpoint_path, kin, mge, args.allow_empty)
    else:
        run_watch_loop(
            run_cfg,
            plot_cfg,
            checkpoint_path,
            kin,
            mge,
            poll_seconds=args.poll_seconds,
            allow_empty=args.allow_empty,
        )


if __name__ == "__main__":
    run_cli()
