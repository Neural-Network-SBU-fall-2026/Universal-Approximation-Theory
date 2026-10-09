"""Reproducible single-hidden-layer sigmoid approximation on [-pi, pi]."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WIDTHS = (4, 8, 16, 32)
SLOPE = 2.0
RCOND = 1e-12


def sigmoid(z):
    """Stable logistic activation, including large positive/negative inputs."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(z)))


def hidden_parameters(width):
    if width < 2:
        raise ValueError("width must be at least two")
    # The first neuron has w=b=0, so sigma(0)=1/2 supplies a constant.
    # This preserves the exact requested sum, without an extra output bias.
    centers = np.linspace(-np.pi, np.pi, width - 1)
    w = np.r_[0.0, np.full(width - 1, SLOPE)]
    b = np.r_[0.0, -SLOPE * centers]
    return w, b


def features(x, w, b):
    return sigmoid(np.asarray(x).reshape(-1, 1) * w + b)


def predict(x, w, b, alpha):
    return features(x, w, b) @ alpha


def fit(x, y, width):
    w, b = hidden_parameters(width)
    matrix = features(x, w, b)
    alpha, _, rank, singular_values = np.linalg.lstsq(matrix, y, rcond=RCOND)
    return w, b, alpha, int(rank), float(singular_values[0] / singular_values[-1])


def metrics(y_true, y_pred):
    error = np.asarray(y_pred) - y_true
    return {
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "r2": float(1 - np.sum(error**2) / np.sum((y_true - np.mean(y_true))**2)),
    }


def run(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train = np.linspace(-np.pi, np.pi, 256)
    # Midpoints of 10,000 cells are disjoint from the 256 training points.
    step = 2 * np.pi / 10000
    x_test = -np.pi + (np.arange(10000) + 0.5) * step
    y_train, y_test = np.sin(x_train), np.sin(x_test)
    endpoints = np.array([-np.pi, np.pi])
    report = {
        "interval": [-float(np.pi), float(np.pi)],
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "activation": "logistic sigmoid",
        "hidden_slope": SLOPE,
        "lstsq_rcond": RCOND,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "models": [],
    }
    models = {}
    for width in WIDTHS:
        w, b, alpha, rank, condition = fit(x_train, y_train, width)
        test_prediction = predict(x_test, w, b, alpha)
        endpoint_error = np.abs(predict(endpoints, w, b, alpha) - np.sin(endpoints))
        entry = {
            "width": width,
            "train": metrics(y_train, predict(x_train, w, b, alpha)),
            "test": metrics(y_test, test_prediction),
            "endpoint_max_abs_error": float(endpoint_error.max()),
            "sampled_interval_max_abs_error": float(max(endpoint_error.max(), np.abs(test_prediction-y_test).max())),
            "effective_rank": rank,
            "design_condition_number": condition,
            "max_abs_output_weight": float(np.abs(alpha).max()),
        }
        report["models"].append(entry)
        models[width] = (w, b, alpha, test_prediction)
        np.savez(output_dir / f"model_{width}.npz", w=w, b=b, alpha=alpha)
        print(f"m={width:2d}: test RMSE={entry['test']['rmse']:.6e}, "
              f"MAE={entry['test']['mae']:.6e}, "
              f"sampled max={entry['sampled_interval_max_abs_error']:.6e}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, width in zip(axes.flat, WIDTHS):
        ax.plot(x_test, y_test, color="black", label="sin(x)")
        ax.plot(x_test, models[width][3], "--", label=f"N(x), m={width}")
        ax.scatter(x_train[::8], y_train[::8], s=12, alpha=0.5, label="Training subset")
        ax.set_title(f"{width} hidden neurons")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x), N(x)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Single-hidden-layer sigmoid approximation on [-pi, pi]")
    fig.tight_layout()
    fig.savefig(output_dir / "approximation.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for width in WIDTHS:
        error = models[width][3] - y_test
        axes[0].plot(x_test, error, label=f"m={width}")
        axes[1].semilogy(x_test, np.maximum(np.abs(error), 1e-16), label=f"m={width}")
    axes[0].set_ylabel("Signed error: N(x) - sin(x)")
    axes[1].set_ylabel("Absolute error (log scale)")
    for ax in axes:
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pointwise_error.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in (("rmse", "Test RMSE"), ("mae", "Test MAE")):
        ax.semilogy(WIDTHS, [row["test"][key] for row in report["models"]], "o-", label=label)
    ax.semilogy(WIDTHS, [row["sampled_interval_max_abs_error"] for row in report["models"]], "s-", label="Sampled interval maximum")
    ax.set_xlabel("Hidden neurons (including constant neuron)")
    ax.set_ylabel("Error (log scale)")
    ax.set_xticks(WIDTHS)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "width_vs_error.png", dpi=160)
    plt.close(fig)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("sine_results"))
    run(parser.parse_args().output_dir)
