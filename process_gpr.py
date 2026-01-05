#!/usr/bin/env python3
import os
import sys
import math
import argparse
from datetime import datetime
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(REPO_ROOT, "3DCUBE_GPR_BA5.txt")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs")

sys.path.insert(0, os.path.join(REPO_ROOT, "irlib-main"))
from irlib.gather import Gather
from irlib.mig_kirchoff import mig_kirchoff


def load_line_data(path, target_line):
    max_trace = 0
    max_sample = 0
    # First pass: determine dimensions for the requested line.
    with open(path, "r", encoding="ascii", errors="ignore") as handle:
        for row in handle:
            parts = row.split()
            if len(parts) != 4:
                continue
            trace_idx = int(parts[0])
            line_idx = int(parts[1])
            sample_idx = int(parts[2])
            if line_idx != target_line:
                continue
            if trace_idx > max_trace:
                max_trace = trace_idx
            if sample_idx > max_sample:
                max_sample = sample_idx

    if max_trace == 0 or max_sample == 0:
        raise RuntimeError(f"No data found for line {target_line}")

    data = np.zeros((max_sample, max_trace), dtype=np.float32)

    # Second pass: populate array for the requested line.
    with open(path, "r", encoding="ascii", errors="ignore") as handle:
        for row in handle:
            parts = row.split()
            if len(parts) != 4:
                continue
            trace_idx = int(parts[0])
            line_idx = int(parts[1])
            sample_idx = int(parts[2])
            if line_idx != target_line:
                continue
            data[sample_idx - 1, trace_idx - 1] = float(parts[3])

    return data


def list_line_ids(path):
    line_ids = set()
    with open(path, "r", encoding="ascii", errors="ignore") as handle:
        for row in handle:
            parts = row.split()
            if len(parts) != 4:
                continue
            line_ids.add(int(parts[1]))
    return sorted(line_ids)

def load_cube_data(path, line_ids):
    max_trace = 0
    max_sample = 0
    line_set = set(line_ids)
    # First pass: determine dimensions for requested lines.
    with open(path, "r", encoding="ascii", errors="ignore") as handle:
        for row in handle:
            parts = row.split()
            if len(parts) != 4:
                continue
            trace_idx = int(parts[0])
            line_idx = int(parts[1])
            sample_idx = int(parts[2])
            if line_idx not in line_set:
                continue
            if trace_idx > max_trace:
                max_trace = trace_idx
            if sample_idx > max_sample:
                max_sample = sample_idx

    if max_trace == 0 or max_sample == 0:
        raise RuntimeError("No data found for requested lines.")

    line_ids_sorted = sorted(line_ids)
    line_index = {line_id: idx for idx, line_id in enumerate(line_ids_sorted)}
    data = np.zeros((max_sample, max_trace, len(line_ids_sorted)), dtype=np.float32)

    # Second pass: populate cube.
    with open(path, "r", encoding="ascii", errors="ignore") as handle:
        for row in handle:
            parts = row.split()
            if len(parts) != 4:
                continue
            trace_idx = int(parts[0])
            line_idx = int(parts[1])
            sample_idx = int(parts[2])
            if line_idx not in line_index:
                continue
            data[sample_idx - 1, trace_idx - 1, line_index[line_idx]] = float(
                parts[3]
            )

    return data, line_ids_sorted


def dc_shift_removal(data):
    return data - np.mean(data, axis=0, keepdims=True)


def dewow(data):
    return dc_shift_removal(data)


def bandpass_fft(data, dt, fmin, fmax, taper_hz=10e6):
    n_samples = data.shape[0]
    freqs = np.fft.rfftfreq(n_samples, dt)
    mask = np.zeros_like(freqs)
    low_start = max(fmin - taper_hz, 0.0)
    low_end = fmin
    high_start = fmax
    high_end = fmax + taper_hz

    passband = (freqs >= low_end) & (freqs <= high_start)
    mask[passband] = 1.0

    if taper_hz > 0:
        low_taper = (freqs >= low_start) & (freqs < low_end)
        if np.any(low_taper):
            x = (freqs[low_taper] - low_start) / (low_end - low_start)
            mask[low_taper] = 0.5 * (1.0 - np.cos(np.pi * x))

        high_taper = (freqs > high_start) & (freqs <= high_end)
        if np.any(high_taper):
            x = (freqs[high_taper] - high_start) / (high_end - high_start)
            mask[high_taper] = 0.5 * (1.0 + np.cos(np.pi * x))
    spectrum = np.fft.rfft(data, axis=0)
    spectrum *= mask[:, None]
    return np.fft.irfft(spectrum, n=n_samples, axis=0)


def plot_frequency_domain(
    data,
    dt,
    title,
    out_path,
    fmax_display=200e6,
    db_floor=-60.0,
):
    n_samples = data.shape[0]
    freqs = np.fft.rfftfreq(n_samples, dt)
    spectrum = np.fft.rfft(data, axis=0)
    amp = np.abs(spectrum)
    max_per_trace = np.max(amp, axis=0)
    max_per_trace[max_per_trace == 0] = 1.0
    amp_norm = amp / max_per_trace
    amp_db = 20.0 * np.log10(np.clip(amp_norm, 1e-12, None))
    mean_amp = np.mean(amp_norm, axis=1)
    mean_db = 20.0 * np.log10(np.clip(mean_amp, 1e-12, None))
    mask = freqs <= fmax_display

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        freqs[mask] / 1e6,
        amp_db[mask, :],
        color="0.6",
        linewidth=0.6,
        alpha=0.35,
    )
    ax.plot(
        freqs[mask] / 1e6,
        mean_db[mask],
        color="red",
        linewidth=1.5,
    )
    ax.set_title(title)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Amplitude (dB re: per-trace max)")
    ax.set_ylim(db_floor, 0.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _pad_sizes(size):
    pad_before = size // 2
    pad_after = size - 1 - pad_before
    return pad_before, pad_after


def mean_filter_xz(data, size=3):
    pad_before, pad_after = _pad_sizes(size)
    padded = np.pad(
        data,
        ((pad_before, pad_after), (pad_before, pad_after)),
        mode="edge",
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return windows.mean(axis=(-1, -2))


def mean_filter_yz(cube, size_y=2, size_z=2):
    pad_z_before, pad_z_after = _pad_sizes(size_z)
    pad_y_before, pad_y_after = _pad_sizes(size_y)
    padded = np.pad(
        cube,
        ((pad_z_before, pad_z_after), (0, 0), (pad_y_before, pad_y_after)),
        mode="edge",
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (size_z, size_y),
        axis=(0, 2),
    )
    return windows.mean(axis=(-1, -2))


def mean_filter_xz_cube(cube, size_x=5, size_z=5):
    pad_z_before, pad_z_after = _pad_sizes(size_z)
    pad_x_before, pad_x_after = _pad_sizes(size_x)
    padded = np.pad(
        cube,
        ((pad_z_before, pad_z_after), (pad_x_before, pad_x_after), (0, 0)),
        mode="edge",
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (size_z, size_x),
        axis=(0, 1),
    )
    return windows.mean(axis=(-1, -2))


def analytic_magnitude(data):
    n_samples = data.shape[0]
    spectrum = np.fft.fft(data, axis=0)
    phase_shift = np.empty_like(spectrum)
    pos_mask = np.arange(n_samples) <= (n_samples // 2 - 1)
    phase_shift[pos_mask, :] = spectrum[pos_mask, :] * np.exp(-1j * np.pi / 2.0)
    phase_shift[~pos_mask, :] = spectrum[~pos_mask, :] * np.exp(1j * np.pi / 2.0)
    hilbert = np.fft.ifft(phase_shift, axis=0).real
    return np.sqrt(data**2 + hilbert**2)


def power_db_max_2d(data):
    power = analytic_magnitude(data)
    max_per_trace = np.max(power, axis=0)
    max_per_trace[max_per_trace == 0.0] = 1.0
    return 10.0 * np.log10((power**2) / (max_per_trace**2))


def apply_mean_xz_cube(cube, size):
    filtered = np.empty_like(cube)
    for line_idx in range(cube.shape[2]):
        filtered[:, :, line_idx] = mean_filter_xz(cube[:, :, line_idx], size=size)
    return filtered


def apply_bandpass_cube(cube, dt, fmin, fmax):
    filtered = np.empty_like(cube)
    for line_idx in range(cube.shape[2]):
        filtered[:, :, line_idx] = bandpass_fft(
            cube[:, :, line_idx], dt, fmin, fmax
        )
    return filtered


def write_fort200(path, data):
    with open(path, "w", encoding="ascii") as handle:
        for row in data:
            handle.write(" ".join(f"{val: .13f}" for val in row) + "\n")


def apply_agc(data, dt, timewin=100e-9):
    gather = Gather(data.copy())
    gather.rate = dt
    gather.DoAutoGainControl(timewin=timewin)
    return gather.data


def horizontal_subtraction(data):
    return data - np.mean(data, axis=1, keepdims=True)


def save_radargrams(
    data,
    dt,
    step_num,
    title,
    zoom_range,
    full_dir,
    zoom_dir,
    agc_ns,
    trace_spacing,
    max_depth,
    file_tag=None,
    apply_agc_flag=False,
):
    if apply_agc_flag:
        data_to_plot = apply_agc(data, dt, timewin=agc_ns * 1e-9)
    else:
        data_to_plot = data
    n_samples = data_to_plot.shape[0]
    z0 = max(0, min(zoom_range[0], n_samples - 1))
    z1 = max(z0 + 1, min(zoom_range[1], n_samples))
    if apply_agc_flag:
        agc_samples = int(round((agc_ns * 1e-9) / dt))
        title_text = (
            f"Step {step_num}: {title} | AGC {agc_ns:.0f} ns "
            f"({agc_samples} samples)"
        )
    else:
        title_text = f"Step {step_num}: {title}"
    title_zoom = (
        f"{title_text} | zoom {zoom_range[0]}-{zoom_range[1]} samples"
    )
    title_size = 11 if len(title_text) < 90 else 9
    title_zoom_size = 11 if len(title_zoom) < 90 else 9
    vmax = np.percentile(np.abs(data_to_plot), 99)

    if file_tag is None:
        file_tag = (
            f"agc_{int(round(agc_ns)):03d}ns" if apply_agc_flag else "raw"
        )

    max_x = (data_to_plot.shape[1] - 1) * trace_spacing
    depth_max = max_depth
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(
        data_to_plot,
        cmap="seismic",
        aspect=2.0,
        vmin=-vmax,
        vmax=vmax,
        extent=(0, max_x, depth_max, 0),
    )
    ax.set_title(title_text, fontsize=title_size)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    fig.tight_layout()
    full_path = os.path.join(
        full_dir, f"step_{step_num:02d}_{file_tag}_full.png"
    )
    fig.savefig(full_path, dpi=200)
    plt.close(fig)

    depth0 = (z0 / max(1, n_samples - 1)) * max_depth
    depth1 = (z1 / max(1, n_samples - 1)) * max_depth
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(
        data_to_plot[z0:z1, :],
        cmap="seismic",
        aspect=2.0,
        vmin=-vmax,
        vmax=vmax,
        extent=(0, max_x, depth1, depth0),
    )
    ax.set_title(title_zoom, fontsize=title_zoom_size)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    fig.tight_layout()
    zoom_path = os.path.join(
        zoom_dir, f"step_{step_num:02d}_{file_tag}_zoom.png"
    )
    fig.savefig(zoom_path, dpi=200)
    plt.close(fig)

    return full_path, zoom_path


def save_compare_radargrams(
    data_a,
    data_b,
    dt,
    step_num,
    title,
    zoom_range,
    full_dir,
    zoom_dir,
    agc_ns,
    trace_spacing,
    max_depth,
    file_tag=None,
    label_a="Procedure 1",
    label_b="Procedure 2",
):
    data_a_agc = apply_agc(data_a, dt, timewin=agc_ns * 1e-9)
    data_b_agc = apply_agc(data_b, dt, timewin=agc_ns * 1e-9)
    n_samples = min(data_a_agc.shape[0], data_b_agc.shape[0])
    z0 = max(0, min(zoom_range[0], n_samples - 1))
    z1 = max(z0 + 1, min(zoom_range[1], n_samples))
    agc_samples = int(round((agc_ns * 1e-9) / dt))
    title_text = (
        f"Step {step_num}: {title} | AGC {agc_ns:.0f} ns ({agc_samples} samples)"
    )
    title_zoom = (
        f"{title_text} | zoom {zoom_range[0]}-{zoom_range[1]} samples"
    )
    title_size = 11 if len(title_text) < 90 else 9
    title_zoom_size = 11 if len(title_zoom) < 90 else 9
    vmax = np.percentile(
        np.abs(np.concatenate([data_a_agc, data_b_agc], axis=1)), 99
    )

    if file_tag is None:
        file_tag = f"agc_{int(round(agc_ns)):03d}ns"

    max_x_a = (data_a_agc.shape[1] - 1) * trace_spacing
    max_x_b = (data_b_agc.shape[1] - 1) * trace_spacing
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    axes[0].imshow(
        data_a_agc,
        cmap="seismic",
        aspect=2.0,
        vmin=-vmax,
        vmax=vmax,
        extent=(0, max_x_a, max_depth, 0),
    )
    axes[0].set_title(f"{label_a}", fontsize=10)
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Depth (m)")
    axes[1].imshow(
        data_b_agc,
        cmap="seismic",
        aspect=2.0,
        vmin=-vmax,
        vmax=vmax,
        extent=(0, max_x_b, max_depth, 0),
    )
    axes[1].set_title(f"{label_b}", fontsize=10)
    axes[1].set_xlabel("Distance (m)")
    fig.suptitle(title_text, fontsize=title_size)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    full_path = os.path.join(
        full_dir,
        f"step_{step_num:02d}_compare_{file_tag}_full.png",
    )
    fig.savefig(full_path, dpi=200)
    plt.close(fig)

    depth0 = (z0 / max(1, n_samples - 1)) * max_depth
    depth1 = (z1 / max(1, n_samples - 1)) * max_depth
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    axes[0].imshow(
        data_a_agc[z0:z1, :],
        cmap="seismic",
        aspect=2.0,
        vmin=-vmax,
        vmax=vmax,
        extent=(0, max_x_a, depth1, depth0),
    )
    axes[0].set_title(f"{label_a}", fontsize=10)
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Depth (m)")
    axes[1].imshow(
        data_b_agc[z0:z1, :],
        cmap="seismic",
        aspect=2.0,
        vmin=-vmax,
        vmax=vmax,
        extent=(0, max_x_b, depth1, depth0),
    )
    axes[1].set_title(f"{label_b}", fontsize=10)
    axes[1].set_xlabel("Distance (m)")
    fig.suptitle(title_zoom, fontsize=title_zoom_size)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    zoom_path = os.path.join(
        zoom_dir,
        f"step_{step_num:02d}_compare_{file_tag}_zoom.png",
    )
    fig.savefig(zoom_path, dpi=200)
    plt.close(fig)

    return full_path, zoom_path


def write_processed_data(handle, data, line_id):
    n_samples, n_traces = data.shape
    for trace_idx in range(n_traces):
        for sample_idx in range(n_samples):
            handle.write(
                f"{trace_idx + 1} {line_id} {sample_idx + 1} "
                f"{data[sample_idx, trace_idx]:.6f}\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--line-start", type=int)
    parser.add_argument("--line-end", type=int)
    parser.add_argument(
        "--line-ids",
        type=str,
        help="Comma-separated list of line ids to process.",
    )
    parser.add_argument("--mean-size-pre", type=int, default=2)
    parser.add_argument("--mean-size-post", type=int, default=5)
    parser.add_argument(
        "--procedure",
        choices=(
            "procedure_1",
            "procedure_2",
            "no_mig_no_bandpass",
            "proc1",
            "proc2",
            "proc3",
            "proc4",
            "proc1m",
            "proc2m",
            "proc3m",
            "proc4m",
            "both",
        ),
        default="no_mig_no_bandpass",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Enable PNG outputs (disabled by default).",
    )
    parser.add_argument(
        "--plot-lines",
        type=str,
        default="29,37",
        help="Comma-separated line ids for PNG/frequency plots.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        help="Output folder tag, e.g. 250102_1.",
    )
    parser.add_argument(
        "--agc-plots",
        action="store_true",
        help="Apply AGC when saving PNGs.",
    )
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    line_ids = list_line_ids(DATA_PATH)
    if args.line_ids:
        requested = []
        for part in args.line_ids.split(","):
            part = part.strip()
            if not part:
                continue
            requested.append(int(part))
        line_ids = [line_id for line_id in line_ids if line_id in requested]
    elif args.line_start is not None or args.line_end is not None:
        start = args.line_start if args.line_start is not None else line_ids[0]
        end = args.line_end if args.line_end is not None else line_ids[-1]
        line_ids = [line_id for line_id in line_ids if start <= line_id <= end]
    plot_lines = set()
    if args.plot_lines:
        for part in args.plot_lines.split(","):
            part = part.strip()
            if part:
                plot_lines.add(int(part))
    args.no_plots = not args.plots
    trace_spacing = 0.25
    max_depth = 50.0
    v = 7.5e7
    tmax = 2.0 * max_depth / v
    agc_windows_ns = [50.0] if args.agc_plots else [0.0]
    zoom_range = (500, 1500)
    date_tag = datetime.now().strftime("%y%m%d")
    run_tag = args.run_tag or f"{date_tag}_1"

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    full_root = os.path.join(OUTPUT_ROOT, run_tag, "full")
    zoom_root = os.path.join(OUTPUT_ROOT, run_tag, "zoom")
    freq_root = os.path.join(OUTPUT_ROOT, run_tag, "frequency")
    notes_root = os.path.join(OUTPUT_ROOT, run_tag, "notes")
    data_root = os.path.join(OUTPUT_ROOT, run_tag, "data")
    if not args.no_plots:
        os.makedirs(freq_root, exist_ok=True)

    output_p1 = os.path.join(REPO_ROOT, "3DCUBE_GPR_BA5_new2Df.txt")
    output_p2 = os.path.join(REPO_ROOT, "3DCUBE_GPR_BA5_wobp.txt")
    output_power = os.path.join(
        data_root,
        "no_mig_no_bandpass",
        "3DCUBE_GPR_BA5_power_db_max.txt",
    )
    proc_outputs = {
        "procedure_1": {"amplitude": output_p1, "power": None},
        "procedure_2": {"amplitude": output_p2, "power": None},
        "no_mig_no_bandpass": {"amplitude": None, "power": output_power},
        "proc1": {
            "amplitude": os.path.join(
                data_root, "proc1", "3DCUBE_GPR_BA5_amplitude.txt"
            ),
            "power": os.path.join(
                data_root, "proc1", "3DCUBE_GPR_BA5_power_db_max.txt"
            ),
        },
        "proc2": {
            "amplitude": os.path.join(
                data_root, "proc2", "3DCUBE_GPR_BA5_amplitude.txt"
            ),
            "power": os.path.join(
                data_root, "proc2", "3DCUBE_GPR_BA5_power_db_max.txt"
            ),
        },
        "proc3": {
            "amplitude": os.path.join(
                data_root, "proc3", "3DCUBE_GPR_BA5_amplitude.txt"
            ),
            "power": os.path.join(
                data_root, "proc3", "3DCUBE_GPR_BA5_power_db_max.txt"
            ),
        },
        "proc4": {
            "amplitude": os.path.join(
                data_root, "proc4", "3DCUBE_GPR_BA5_amplitude.txt"
            ),
            "power": os.path.join(
                data_root, "proc4", "3DCUBE_GPR_BA5_power_db_max.txt"
            ),
        },
        "proc1m": {
            "amplitude_xwin1": os.path.join(
                data_root,
                "proc1m",
                "xwindow_1m",
                "3DCUBE_GPR_BA5_amplitude.txt",
            ),
            "power_xwin1": os.path.join(
                data_root,
                "proc1m",
                "xwindow_1m",
                "3DCUBE_GPR_BA5_power_db_max.txt",
            ),
            "amplitude_xwin5": os.path.join(
                data_root,
                "proc1m",
                "xwindow_5m",
                "3DCUBE_GPR_BA5_amplitude.txt",
            ),
            "power_xwin5": os.path.join(
                data_root,
                "proc1m",
                "xwindow_5m",
                "3DCUBE_GPR_BA5_power_db_max.txt",
            ),
        },
        "proc2m": {
            "amplitude_xwin1": os.path.join(
                data_root,
                "proc2m",
                "xwindow_1m",
                "3DCUBE_GPR_BA5_amplitude.txt",
            ),
            "power_xwin1": os.path.join(
                data_root,
                "proc2m",
                "xwindow_1m",
                "3DCUBE_GPR_BA5_power_db_max.txt",
            ),
            "amplitude_xwin5": os.path.join(
                data_root,
                "proc2m",
                "xwindow_5m",
                "3DCUBE_GPR_BA5_amplitude.txt",
            ),
            "power_xwin5": os.path.join(
                data_root,
                "proc2m",
                "xwindow_5m",
                "3DCUBE_GPR_BA5_power_db_max.txt",
            ),
        },
        "proc3m": {
            "amplitude_xwin1": os.path.join(
                data_root,
                "proc3m",
                "xwindow_1m",
                "3DCUBE_GPR_BA5_amplitude.txt",
            ),
            "power_xwin1": os.path.join(
                data_root,
                "proc3m",
                "xwindow_1m",
                "3DCUBE_GPR_BA5_power_db_max.txt",
            ),
            "amplitude_xwin5": os.path.join(
                data_root,
                "proc3m",
                "xwindow_5m",
                "3DCUBE_GPR_BA5_amplitude.txt",
            ),
            "power_xwin5": os.path.join(
                data_root,
                "proc3m",
                "xwindow_5m",
                "3DCUBE_GPR_BA5_power_db_max.txt",
            ),
        },
        "proc4m": {
            "amplitude_xwin1": os.path.join(
                data_root,
                "proc4m",
                "xwindow_1m",
                "3DCUBE_GPR_BA5_amplitude.txt",
            ),
            "power_xwin1": os.path.join(
                data_root,
                "proc4m",
                "xwindow_1m",
                "3DCUBE_GPR_BA5_power_db_max.txt",
            ),
            "amplitude_xwin5": os.path.join(
                data_root,
                "proc4m",
                "xwindow_5m",
                "3DCUBE_GPR_BA5_amplitude.txt",
            ),
            "power_xwin5": os.path.join(
                data_root,
                "proc4m",
                "xwindow_5m",
                "3DCUBE_GPR_BA5_power_db_max.txt",
            ),
        },
    }

    procedures_all = [
        (
            "procedure_1",
            "DC-shift removal -> 2-D mean filter (YxZ 2x2) -> band-pass -> "
            "2-D mean filter (XxZ 5x5) -> migration",
        ),
        (
            "procedure_2",
            "DC-shift removal -> band-pass -> 2-D mean filter -> migration",
        ),
        (
            "no_mig_no_bandpass",
            "DeWow -> 2-D mean filter (XxZ 5x5) -> 2-D mean filter (YxZ 2x2)"
            " -> power dB max",
        ),
        (
            "proc1",
            "DeWow -> 2-D mean filter (XxZ 5x5) -> 2-D mean filter (YxZ 2x2)"
            " -> amplitude + power dB max",
        ),
        (
            "proc2",
            "DeWow -> 2-D mean filter (XxZ 5x5) -> amplitude + power dB max",
        ),
        (
            "proc3",
            "DeWow -> band-pass (40-80 MHz) -> 2-D mean filter (XxZ 5x5)"
            " -> amplitude + power dB max",
        ),
        (
            "proc4",
            "DeWow -> band-pass (40-80 MHz) -> 2-D mean filter (XxZ 5x5)"
            " -> 2-D mean filter (YxZ 2x2) -> amplitude + power dB max",
        ),
        (
            "proc1m",
            "DeWow -> 2-D mean filter (XxZ 5x5) -> 2-D mean filter (YxZ 2x2)"
            " -> migration (1 m and 5 m) -> amplitude + power dB max",
        ),
        (
            "proc2m",
            "DeWow -> 2-D mean filter (XxZ 5x5)"
            " -> migration (1 m and 5 m) -> amplitude + power dB max",
        ),
        (
            "proc3m",
            "DeWow -> band-pass (40-80 MHz) -> 2-D mean filter (XxZ 5x5)"
            " -> migration (1 m and 5 m) -> amplitude + power dB max",
        ),
        (
            "proc4m",
            "DeWow -> band-pass (40-80 MHz) -> 2-D mean filter (XxZ 5x5)"
            " -> 2-D mean filter (YxZ 2x2)"
            " -> migration (1 m and 5 m) -> amplitude + power dB max",
        ),
    ]
    if args.procedure == "both":
        procedures = procedures_all
    else:
        procedures = [
            item for item in procedures_all if item[0] == args.procedure
        ]

    data_cube, line_ids = load_cube_data(DATA_PATH, line_ids)
    line_index = {line_id: idx for idx, line_id in enumerate(line_ids)}
    dt = tmax / data_cube.shape[0]
    proc1_enabled = any(name == "procedure_1" for name, _ in procedures)
    power_enabled = any(
        name == "no_mig_no_bandpass" for name, _ in procedures
    )
    proc1_mean_yz = None
    if proc1_enabled:
        proc1_dc = dc_shift_removal(data_cube)
        proc1_mean_yz = mean_filter_yz(
            proc1_dc,
            size_y=args.mean_size_pre,
            size_z=args.mean_size_pre,
        )
        del proc1_dc

    power_dewow = None
    power_xz = None
    power_yz = None
    if power_enabled:
        power_dewow = dewow(data_cube)
        power_xz = apply_mean_xz_cube(power_dewow, args.mean_size_post)
        power_yz = mean_filter_yz(
            power_xz, size_y=args.mean_size_pre, size_z=args.mean_size_pre
        )

    new_proc_names = {"proc1", "proc2", "proc3", "proc4"}
    new_mig_names = {"proc1m", "proc2m", "proc3m", "proc4m"}
    selected = {name for name, _ in procedures}
    new_enabled = bool((new_proc_names | new_mig_names) & selected)
    bandpass_enabled = bool({"proc3", "proc4", "proc3m", "proc4m"} & selected)
    mean_yz_enabled = bool({"proc1", "proc4", "proc1m", "proc4m"} & selected)
    mean_xz_enabled = bool(
        {"proc1", "proc2", "proc3", "proc4", "proc1m", "proc2m", "proc3m", "proc4m"}
        & selected
    )
    new_dewow = None
    new_bandpass = None
    new_mean_xz = None
    new_mean_yz = None
    if new_enabled:
        new_dewow = dewow(data_cube)
        if bandpass_enabled:
            new_bandpass = apply_bandpass_cube(
                new_dewow, dt, 40e6, 80e6
            )
        if mean_xz_enabled:
            source = new_bandpass if bandpass_enabled else new_dewow
            new_mean_xz = apply_mean_xz_cube(source, args.mean_size_post)
        if mean_yz_enabled:
            new_mean_yz = mean_filter_yz(
                new_mean_xz,
                size_y=args.mean_size_pre,
                size_z=args.mean_size_pre,
            )

    output_mode = "a" if args.append else "w"
    output_handles = {}
    for proc_name, _ in procedures:
        outputs = proc_outputs.get(proc_name, {})
        output_handles[proc_name] = {}
        for kind, out_path in outputs.items():
            if out_path:
                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                output_handles[proc_name][kind] = open(
                    out_path, output_mode, encoding="ascii"
                )
    note_written = set()
    note_line_list = ", ".join(f"{line_id}" for line_id in line_ids)
    freq_lines = set(plot_lines)
    try:
        for line_id in line_ids:
            line_idx = line_index[line_id]
            data = data_cube[:, :, line_idx]
            final_steps = {}
            do_plots_line = (not args.no_plots) and (line_id in plot_lines)

            for proc_name, proc_label in procedures:
                steps = []
                if proc_name in new_mig_names:
                    line_full_dir = None
                    line_zoom_dir = None
                else:
                    line_full_dir = os.path.join(
                        full_root, proc_name, f"line_{line_id:02d}"
                    )
                    line_zoom_dir = os.path.join(
                        zoom_root, proc_name, f"line_{line_id:02d}"
                    )
                    os.makedirs(line_full_dir, exist_ok=True)
                    os.makedirs(line_zoom_dir, exist_ok=True)
                note_path = os.path.join(
                    notes_root,
                    f"processing_notes_{run_tag}_{proc_name}.txt",
                )

                if proc_name == "procedure_1":
                    # Step 1: DC shift removal.
                    step_data = dc_shift_removal(data)
                    steps.append(("after DC-shift removal", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                1,
                                "after DC-shift removal",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                    if do_plots_line and line_id in freq_lines:
                        line_freq_dir = os.path.join(
                            freq_root,
                            proc_name,
                            f"line_{line_id:02d}",
                        )
                        os.makedirs(line_freq_dir, exist_ok=True)
                        plot_frequency_domain(
                            step_data,
                            dt,
                            f"Line {line_id:02d} after DeWow (0-200 MHz)",
                            os.path.join(
                                line_freq_dir,
                                "step_01_after_dewow.png",
                            ),
                        )
                    # Step 2: 2-D mean filter (YxZ).
                    step_data = proc1_mean_yz[:, :, line_idx]
                    steps.append(("after 2-D mean filter (YxZ 2x2)", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                2,
                                "after 2-D mean filter (YxZ 2x2)",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                    if do_plots_line and line_id in freq_lines:
                        line_freq_dir = os.path.join(
                            freq_root,
                            proc_name,
                            f"line_{line_id:02d}",
                        )
                        os.makedirs(line_freq_dir, exist_ok=True)
                        plot_frequency_domain(
                            step_data,
                            dt,
                            f"Line {line_id:02d} after mean filter XxZ (0-200 MHz)",
                            os.path.join(
                                line_freq_dir,
                                "step_02_after_mean_xz.png",
                            ),
                        )
                    # Step 3: Band-pass filter 10-100 MHz.
                    if do_plots_line:
                        plot_frequency_domain(
                            step_data,
                            dt,
                            "Before band-pass (10-100 MHz)",
                            os.path.join(
                                freq_root,
                                f"line_{line_id:02d}_{proc_name}_before_bandpass.png",
                            ),
                        )
                    step_data = bandpass_fft(step_data, dt, 10e6, 100e6)
                    steps.append(("after band-pass (10-100 MHz)", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                3,
                                "after band-pass (10-100 MHz)",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                    if do_plots_line and line_id in freq_lines:
                        line_freq_dir = os.path.join(
                            freq_root,
                            proc_name,
                            f"line_{line_id:02d}",
                        )
                        os.makedirs(line_freq_dir, exist_ok=True)
                        plot_frequency_domain(
                            step_data,
                            dt,
                            f"Line {line_id:02d} after mean filter YxZ (0-200 MHz)",
                            os.path.join(
                                line_freq_dir,
                                "step_03_after_mean_yz.png",
                            ),
                        )
                    if do_plots_line:
                        plot_frequency_domain(
                            step_data,
                            dt,
                            "After band-pass (10-100 MHz)",
                            os.path.join(
                                freq_root,
                                f"line_{line_id:02d}_{proc_name}_after_bandpass.png",
                            ),
                        )
                    # Step 4: 2-D mean filter (XxZ).
                    step_data = mean_filter_xz(step_data, size=args.mean_size_post)
                    steps.append(("after 2-D mean filter (XxZ 5x5)", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                4,
                                "after 2-D mean filter (XxZ 5x5)",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                    if do_plots_line and line_id in freq_lines:
                        line_freq_dir = os.path.join(
                            freq_root,
                            proc_name,
                            f"line_{line_id:02d}",
                        )
                        os.makedirs(line_freq_dir, exist_ok=True)
                        plot_frequency_domain(
                            step_data,
                            dt,
                            f"Line {line_id:02d} after power dB max (0-200 MHz)",
                            os.path.join(
                                line_freq_dir,
                                "step_04_after_power_db_max.png",
                            ),
                        )
                elif proc_name == "procedure_2":
                    # Step 1: DC shift removal.
                    step_data = dc_shift_removal(data)
                    steps.append(("after DC-shift removal", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                1,
                                "after DC-shift removal",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                    # Step 2: Band-pass filter 10-100 MHz.
                    if do_plots_line:
                        plot_frequency_domain(
                            step_data,
                            dt,
                            "Before band-pass (10-100 MHz)",
                            os.path.join(
                                freq_root,
                                f"line_{line_id:02d}_{proc_name}_before_bandpass.png",
                            ),
                        )
                    step_data = bandpass_fft(step_data, dt, 10e6, 100e6)
                    steps.append(("after band-pass (10-100 MHz)", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                2,
                                "after band-pass (10-100 MHz)",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                    if do_plots_line:
                        plot_frequency_domain(
                            step_data,
                            dt,
                            "After band-pass (10-100 MHz)",
                            os.path.join(
                                freq_root,
                                f"line_{line_id:02d}_{proc_name}_after_bandpass.png",
                            ),
                        )
                    # Step 3: 2-D mean filter (legacy size).
                    step_data = mean_filter_xz(step_data, size=args.mean_size_post)
                    steps.append(("after 2-D mean filter", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                3,
                                "after 2-D mean filter",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )

                elif proc_name in new_proc_names:
                    # Step 1: DeWow.
                    step_data = new_dewow[:, :, line_idx]
                    steps.append(("after DeWow", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                1,
                                "after DeWow",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                    if proc_name in ("proc3", "proc4") and do_plots_line:
                        line_freq_dir = os.path.join(
                            freq_root,
                            proc_name,
                            f"line_{line_id:02d}",
                        )
                        os.makedirs(line_freq_dir, exist_ok=True)
                        plot_frequency_domain(
                            step_data,
                            dt,
                            "Before band-pass (40-80 MHz)",
                            os.path.join(
                                line_freq_dir,
                                "step_01_before_bandpass.png",
                            ),
                        )
                    # Step 2: band-pass for proc3/proc4.
                    if proc_name in ("proc3", "proc4"):
                        step_data = new_bandpass[:, :, line_idx]
                        steps.append(("after band-pass (40-80 MHz)", step_data))
                        if do_plots_line:
                            for agc_ns in agc_windows_ns:
                                save_radargrams(
                                    step_data,
                                    dt,
                                    2,
                                    "after band-pass (40-80 MHz)",
                                    zoom_range,
                                    line_full_dir,
                                    line_zoom_dir,
                                    agc_ns,
                                    trace_spacing,
                                    max_depth,
                                    apply_agc_flag=args.agc_plots,
                                )
                            line_freq_dir = os.path.join(
                                freq_root,
                                proc_name,
                                f"line_{line_id:02d}",
                            )
                            os.makedirs(line_freq_dir, exist_ok=True)
                            plot_frequency_domain(
                                step_data,
                                dt,
                                "After band-pass (40-80 MHz)",
                                os.path.join(
                                    line_freq_dir,
                                    "step_02_after_bandpass.png",
                                ),
                            )
                    # Step 3: 2-D mean filter (XxZ).
                    step_data = new_mean_xz[:, :, line_idx]
                    step_num = 2 if proc_name in ("proc1", "proc2") else 3
                    steps.append(("after 2-D mean filter (XxZ 5x5)", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                step_num,
                                "after 2-D mean filter (XxZ 5x5)",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                    # Step 4: optional 2-D mean filter (YxZ).
                    if proc_name in ("proc1", "proc4"):
                        step_data = new_mean_yz[:, :, line_idx]
                        step_num = 3 if proc_name == "proc1" else 4
                        steps.append(
                            ("after 2-D mean filter (YxZ 2x2)", step_data)
                        )
                        if do_plots_line:
                            for agc_ns in agc_windows_ns:
                                save_radargrams(
                                    step_data,
                                    dt,
                                    step_num,
                                    "after 2-D mean filter (YxZ 2x2)",
                                    zoom_range,
                                    line_full_dir,
                                    line_zoom_dir,
                                    agc_ns,
                                    trace_spacing,
                                    max_depth,
                                    apply_agc_flag=args.agc_plots,
                                )
                    # Final: amplitude and power dB max.
                    amp_data = step_data
                    power_data = power_db_max_2d(amp_data)
                    steps.append(("after power dB max", power_data))
                    if do_plots_line:
                        step_num = 4 if proc_name == "proc1" else 3
                        if proc_name == "proc3":
                            step_num = 4
                        if proc_name == "proc4":
                            step_num = 5
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                power_data,
                                dt,
                                step_num,
                                "after power dB max",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                elif proc_name in new_mig_names:
                    # Step 1: DeWow.
                    step_data = new_dewow[:, :, line_idx]
                    steps.append(("after DeWow", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            line_full_dir = os.path.join(
                                full_root,
                                proc_name,
                                "xwindow_1m",
                                f"line_{line_id:02d}",
                            )
                            line_zoom_dir = os.path.join(
                                zoom_root,
                                proc_name,
                                "xwindow_1m",
                                f"line_{line_id:02d}",
                            )
                            os.makedirs(line_full_dir, exist_ok=True)
                            os.makedirs(line_zoom_dir, exist_ok=True)
                            save_radargrams(
                                step_data,
                                dt,
                                1,
                                "after DeWow",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                            line_full_dir = os.path.join(
                                full_root,
                                proc_name,
                                "xwindow_5m",
                                f"line_{line_id:02d}",
                            )
                            line_zoom_dir = os.path.join(
                                zoom_root,
                                proc_name,
                                "xwindow_5m",
                                f"line_{line_id:02d}",
                            )
                            os.makedirs(line_full_dir, exist_ok=True)
                            os.makedirs(line_zoom_dir, exist_ok=True)
                            save_radargrams(
                                step_data,
                                dt,
                                1,
                                "after DeWow",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                    if proc_name in ("proc3m", "proc4m") and do_plots_line:
                        line_freq_dir = os.path.join(
                            freq_root,
                            proc_name,
                            f"line_{line_id:02d}",
                        )
                        os.makedirs(line_freq_dir, exist_ok=True)
                        plot_frequency_domain(
                            step_data,
                            dt,
                            "Before band-pass (40-80 MHz)",
                            os.path.join(
                                line_freq_dir,
                                "step_01_before_bandpass.png",
                            ),
                        )
                    # Step 2: band-pass for proc3m/proc4m.
                    if proc_name in ("proc3m", "proc4m"):
                        step_data = new_bandpass[:, :, line_idx]
                        steps.append(("after band-pass (40-80 MHz)", step_data))
                        if do_plots_line:
                            for agc_ns in agc_windows_ns:
                                line_full_dir = os.path.join(
                                    full_root,
                                    proc_name,
                                    "xwindow_1m",
                                    f"line_{line_id:02d}",
                                )
                                line_zoom_dir = os.path.join(
                                    zoom_root,
                                    proc_name,
                                    "xwindow_1m",
                                    f"line_{line_id:02d}",
                                )
                                os.makedirs(line_full_dir, exist_ok=True)
                                os.makedirs(line_zoom_dir, exist_ok=True)
                                save_radargrams(
                                    step_data,
                                    dt,
                                    2,
                                    "after band-pass (40-80 MHz)",
                                    zoom_range,
                                    line_full_dir,
                                    line_zoom_dir,
                                    agc_ns,
                                    trace_spacing,
                                    max_depth,
                                    apply_agc_flag=args.agc_plots,
                                )
                                line_full_dir = os.path.join(
                                    full_root,
                                    proc_name,
                                    "xwindow_5m",
                                    f"line_{line_id:02d}",
                                )
                                line_zoom_dir = os.path.join(
                                    zoom_root,
                                    proc_name,
                                    "xwindow_5m",
                                    f"line_{line_id:02d}",
                                )
                                os.makedirs(line_full_dir, exist_ok=True)
                                os.makedirs(line_zoom_dir, exist_ok=True)
                                save_radargrams(
                                    step_data,
                                    dt,
                                    2,
                                    "after band-pass (40-80 MHz)",
                                    zoom_range,
                                    line_full_dir,
                                    line_zoom_dir,
                                    agc_ns,
                                    trace_spacing,
                                    max_depth,
                                    apply_agc_flag=args.agc_plots,
                                )
                            line_freq_dir = os.path.join(
                                freq_root,
                                proc_name,
                                f"line_{line_id:02d}",
                            )
                            os.makedirs(line_freq_dir, exist_ok=True)
                            plot_frequency_domain(
                                step_data,
                                dt,
                                "After band-pass (40-80 MHz)",
                                os.path.join(
                                    line_freq_dir,
                                    "step_02_after_bandpass.png",
                                ),
                            )
                    # Step 3: 2-D mean filter (XxZ).
                    step_data = new_mean_xz[:, :, line_idx]
                    step_num = 2 if proc_name in ("proc1m", "proc2m") else 3
                    steps.append(("after 2-D mean filter (XxZ 5x5)", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            for xwin in ("xwindow_1m", "xwindow_5m"):
                                line_full_dir = os.path.join(
                                    full_root,
                                    proc_name,
                                    xwin,
                                    f"line_{line_id:02d}",
                                )
                                line_zoom_dir = os.path.join(
                                    zoom_root,
                                    proc_name,
                                    xwin,
                                    f"line_{line_id:02d}",
                                )
                                os.makedirs(line_full_dir, exist_ok=True)
                                os.makedirs(line_zoom_dir, exist_ok=True)
                                save_radargrams(
                                    step_data,
                                    dt,
                                    step_num,
                                    "after 2-D mean filter (XxZ 5x5)",
                                    zoom_range,
                                    line_full_dir,
                                    line_zoom_dir,
                                    agc_ns,
                                    trace_spacing,
                                    max_depth,
                                    apply_agc_flag=args.agc_plots,
                                )
                    # Step 4: optional 2-D mean filter (YxZ).
                    if proc_name in ("proc1m", "proc4m"):
                        step_data = new_mean_yz[:, :, line_idx]
                        step_num = 3 if proc_name == "proc1m" else 4
                        steps.append(
                            ("after 2-D mean filter (YxZ 2x2)", step_data)
                        )
                        if do_plots_line:
                            for agc_ns in agc_windows_ns:
                                for xwin in ("xwindow_1m", "xwindow_5m"):
                                    line_full_dir = os.path.join(
                                        full_root,
                                        proc_name,
                                        xwin,
                                        f"line_{line_id:02d}",
                                    )
                                    line_zoom_dir = os.path.join(
                                        zoom_root,
                                        proc_name,
                                        xwin,
                                        f"line_{line_id:02d}",
                                    )
                                    os.makedirs(line_full_dir, exist_ok=True)
                                    os.makedirs(line_zoom_dir, exist_ok=True)
                                    save_radargrams(
                                        step_data,
                                        dt,
                                        step_num,
                                        "after 2-D mean filter (YxZ 2x2)",
                                        zoom_range,
                                        line_full_dir,
                                        line_zoom_dir,
                                        agc_ns,
                                        trace_spacing,
                                        max_depth,
                                        apply_agc_flag=args.agc_plots,
                                    )
                    # Migration for 1 m and 5 m apertures.
                    x = np.arange(step_data.shape[1]) * trace_spacing
                    t = np.arange(step_data.shape[0]) * dt
                    mig_outputs = {}
                    for aperture in (1.0, 5.0):
                        mig_data = mig_kirchoff(
                            step_data,
                            x,
                            t,
                            v=v,
                            xoffset=0.0,
                            xwindow=aperture,
                            pad_spacing=trace_spacing,
                        )
                        power_data = power_db_max_2d(mig_data)
                        mig_outputs[aperture] = (mig_data, power_data)
                        if do_plots_line:
                            step_base = 4 if proc_name in ("proc1m", "proc2m") else 5
                            if proc_name == "proc3m":
                                step_base = 4
                            if proc_name == "proc4m":
                                step_base = 5
                            step_idx = step_base if aperture == 1.0 else step_base + 2
                            line_full_dir = os.path.join(
                                full_root,
                                proc_name,
                                f"xwindow_{int(aperture)}m",
                                f"line_{line_id:02d}",
                            )
                            line_zoom_dir = os.path.join(
                                zoom_root,
                                proc_name,
                                f"xwindow_{int(aperture)}m",
                                f"line_{line_id:02d}",
                            )
                            os.makedirs(line_full_dir, exist_ok=True)
                            os.makedirs(line_zoom_dir, exist_ok=True)
                            for agc_ns in agc_windows_ns:
                                save_radargrams(
                                    mig_data,
                                    dt,
                                    step_idx,
                                    f"after migration (xwindow {aperture:.0f} m)",
                                    zoom_range,
                                    line_full_dir,
                                    line_zoom_dir,
                                    agc_ns,
                                    trace_spacing,
                                    max_depth,
                                    apply_agc_flag=args.agc_plots,
                                )
                                save_radargrams(
                                    power_data,
                                    dt,
                                    step_idx + 1,
                                    f"after power dB max (xwindow {aperture:.0f} m)",
                                    zoom_range,
                                    line_full_dir,
                                    line_zoom_dir,
                                    agc_ns,
                                    trace_spacing,
                                    max_depth,
                                    apply_agc_flag=args.agc_plots,
                                )
                elif proc_name in new_mig_names:
                    for aperture, (amp_data, power_data) in mig_outputs.items():
                        amp_handle = output_handles[proc_name].get(
                            f"amplitude_xwin{int(aperture)}"
                        )
                        power_handle = output_handles[proc_name].get(
                            f"power_xwin{int(aperture)}"
                        )
                        if amp_handle:
                            write_processed_data(amp_handle, amp_data, line_id)
                        if power_handle:
                            write_processed_data(power_handle, power_data, line_id)
                        if line_id in (29, 37):
                            out_dir = os.path.join(
                                data_root,
                                proc_name,
                                f"xwindow_{int(aperture)}m",
                            )
                            write_fort200(
                                os.path.join(
                                    out_dir,
                                    f"line_{line_id:02d}_amplitude_fort.200.txt",
                                ),
                                amp_data,
                            )
                            write_fort200(
                                os.path.join(
                                    out_dir,
                                    f"line_{line_id:02d}_power_db_max_fort.200.txt",
                                ),
                                power_data,
                            )
                else:
                    # Step 1: DeWow.
                    step_data = power_dewow[:, :, line_idx]
                    steps.append(("after DeWow", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                1,
                                "after DeWow",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                    # Step 2: 2-D mean filter (XxZ).
                    step_data = power_xz[:, :, line_idx]
                    steps.append(("after 2-D mean filter (XxZ 5x5)", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                2,
                                "after 2-D mean filter (XxZ 5x5)",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                    # Step 3: 2-D mean filter (YxZ).
                    step_data = power_yz[:, :, line_idx]
                    steps.append(("after 2-D mean filter (YxZ 2x2)", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                3,
                                "after 2-D mean filter (YxZ 2x2)",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                    # Step 4: power dB max.
                    step_data = power_db_max_2d(step_data)
                    steps.append(("after power dB max", step_data))
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                4,
                                "after power dB max",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )

                if proc_name in ("procedure_1", "procedure_2"):
                    # Step 5/4: Kirchhoff migration.
                    x = np.arange(step_data.shape[1]) * trace_spacing
                    t = np.arange(step_data.shape[0]) * dt
                    step_data = mig_kirchoff(
                        step_data,
                        x,
                        t,
                        v=v,
                        xoffset=0.0,
                        xwindow=1.0,
                        pad_spacing=trace_spacing,
                    )
                    steps.append(("after Kirchhoff migration", step_data))
                    step_num = 5 if proc_name == "procedure_1" else 4
                    if do_plots_line:
                        for agc_ns in agc_windows_ns:
                            save_radargrams(
                                step_data,
                                dt,
                                step_num,
                                "after Kirchhoff migration",
                                zoom_range,
                                line_full_dir,
                                line_zoom_dir,
                                agc_ns,
                                trace_spacing,
                                max_depth,
                                apply_agc_flag=args.agc_plots,
                            )
                    final_steps[proc_name] = step_data

                    handle = output_handles[proc_name].get("amplitude")
                    if handle:
                        write_processed_data(handle, step_data, line_id)
                elif proc_name in new_proc_names:
                    amp_handle = output_handles[proc_name].get("amplitude")
                    power_handle = output_handles[proc_name].get("power")
                    if amp_handle:
                        write_processed_data(amp_handle, amp_data, line_id)
                    if power_handle:
                        write_processed_data(power_handle, power_data, line_id)
                    if line_id in (29, 37):
                        out_dir = os.path.join(data_root, proc_name)
                        write_fort200(
                            os.path.join(
                                out_dir,
                                f"line_{line_id:02d}_amplitude_fort.200.txt",
                            ),
                            amp_data,
                        )
                        write_fort200(
                            os.path.join(
                                out_dir,
                                f"line_{line_id:02d}_power_db_max_fort.200.txt",
                            ),
                            power_data,
                        )
                elif proc_name in new_mig_names:
                    for aperture, (amp_data, power_data) in mig_outputs.items():
                        amp_handle = output_handles[proc_name].get(
                            f"amplitude_xwin{int(aperture)}"
                        )
                        power_handle = output_handles[proc_name].get(
                            f"power_xwin{int(aperture)}"
                        )
                        if amp_handle:
                            write_processed_data(amp_handle, amp_data, line_id)
                        if power_handle:
                            write_processed_data(power_handle, power_data, line_id)
                        if line_id in (29, 37):
                            out_dir = os.path.join(
                                data_root,
                                proc_name,
                                f"xwindow_{int(aperture)}m",
                            )
                            os.makedirs(out_dir, exist_ok=True)
                            write_fort200(
                                os.path.join(
                                    out_dir,
                                    f"line_{line_id:02d}_amplitude_fort.200.txt",
                                ),
                                amp_data,
                            )
                            write_fort200(
                                os.path.join(
                                    out_dir,
                                    f"line_{line_id:02d}_power_db_max_fort.200.txt",
                                ),
                                power_data,
                            )
                else:
                    handle = output_handles[proc_name].get("power")
                    if handle:
                        write_processed_data(handle, step_data, line_id)
                    if line_id in (29, 37):
                        fort_path = os.path.join(
                            data_root,
                            "no_mig_no_bandpass",
                            f"line_{line_id:02d}_fort.200.txt",
                        )
                        os.makedirs(os.path.dirname(fort_path), exist_ok=True)
                        write_fort200(fort_path, step_data)

                if proc_name not in note_written:
                    os.makedirs(notes_root, exist_ok=True)
                    with open(note_path, "w", encoding="ascii") as note:
                        note.write("GPR processing sequence for 3DCUBE_GPR_BA5.txt\n")
                        note.write(f"Lines processed: {note_line_list}\n")
                        note.write(f"Procedure: {proc_label}\n")
                        outputs = proc_outputs.get(proc_name, {})
                        for kind, path in outputs.items():
                            if path:
                                note.write(f"{kind} output: {path}\n")
                        note.write("Trace spacing: 0.25 m, line spacing: 0.5 m\n")
                        note.write(f"Samples per trace: {data.shape[0]}\n")
                        if proc_name in ("procedure_1", "procedure_2"):
                            note.write(
                                "Velocity used for migration: "
                                f"{v} m/s (epsilon_r=16)\n"
                            )
                        note.write(f"Max depth: {max_depth} m, dt: {dt:.3e} s\n")
                        if args.agc_plots:
                            agc_parts = []
                            for agc_ns in agc_windows_ns:
                                agc_samples = int(round((agc_ns * 1e-9) / dt))
                                agc_parts.append(
                                    f"{agc_ns:.0f} ns ({agc_samples} samples)"
                                )
                            note.write(
                                "AGC for plots: "
                                + ", ".join(agc_parts)
                                + "\n"
                            )
                        note.write("AGC applied to saved data: none.\n")
                        note.write("\nProcessing steps:\n")
                        if proc_name == "no_mig_no_bandpass":
                            note.write("1) DeWow: subtract mean per trace.\n")
                        elif proc_name in new_proc_names:
                            note.write("1) DeWow: subtract mean per trace.\n")
                        elif proc_name in new_mig_names:
                            note.write("1) DeWow: subtract mean per trace.\n")
                            if proc_name in ("proc3m", "proc4m"):
                                note.write(
                                    "2) Band-pass: FFT mask 40-80 MHz on each trace.\n"
                                )
                                note.write(
                                    f"3) 2-D mean filter (XxZ): {args.mean_size_post}x"
                                    f"{args.mean_size_post} boxcar window.\n"
                                )
                                step_idx = 4
                                if proc_name == "proc4m":
                                    note.write(
                                        f"4) 2-D mean filter (YxZ): {args.mean_size_pre}x"
                                        f"{args.mean_size_pre} boxcar window.\n"
                                    )
                                    step_idx = 5
                                note.write(
                                    f"{step_idx}) Kirchhoff migration: "
                                    "xwindow=1 m and 5 m, xoffset=0 m, pad_spacing=0.25 m.\n"
                                )
                                note.write(
                                    f"{step_idx + 1}) power dB max: Hilbert magnitude.\n"
                                )
                            else:
                                note.write(
                                    f"2) 2-D mean filter (XxZ): {args.mean_size_post}x"
                                    f"{args.mean_size_post} boxcar window.\n"
                                )
                                step_idx = 3
                                if proc_name == "proc1m":
                                    note.write(
                                        f"3) 2-D mean filter (YxZ): {args.mean_size_pre}x"
                                        f"{args.mean_size_pre} boxcar window.\n"
                                    )
                                    step_idx = 4
                                note.write(
                                    f"{step_idx}) Kirchhoff migration: "
                                    "xwindow=1 m and 5 m, xoffset=0 m, pad_spacing=0.25 m.\n"
                                )
                                note.write(
                                    f"{step_idx + 1}) power dB max: Hilbert magnitude.\n"
                                )
                        else:
                            note.write(
                                "1) DC-shift removal: subtract mean per trace.\n"
                            )
                        if proc_name == "procedure_1":
                            note.write(
                                f"2) 2-D mean filter (YxZ): {args.mean_size_pre}x"
                                f"{args.mean_size_pre} boxcar window.\n"
                            )
                            note.write(
                                "3) Band-pass: FFT mask 10-100 MHz on each trace.\n"
                            )
                            note.write(
                                f"4) 2-D mean filter (XxZ): {args.mean_size_post}x"
                                f"{args.mean_size_post} boxcar window.\n"
                            )
                            note.write(
                                "5) Kirchhoff migration: "
                                "irlib.mig_kirchoff.mig_kirchoff\n"
                            )
                            note.write(
                                "   - xwindow=1 m, xoffset=0 m, pad_spacing=0.25 m.\n"
                            )
                        elif proc_name == "procedure_2":
                            note.write(
                                "2) Band-pass: FFT mask 10-100 MHz on each trace.\n"
                            )
                            note.write(
                                f"3) 2-D mean filter: {args.mean_size_post}x"
                                f"{args.mean_size_post} "
                                "boxcar window.\n"
                            )
                            note.write(
                                "4) Kirchhoff migration: "
                                "irlib.mig_kirchoff.mig_kirchoff\n"
                            )
                            note.write(
                                "   - xwindow=1 m, xoffset=0 m, pad_spacing=0.25 m.\n"
                            )
                        elif proc_name in new_proc_names:
                            if proc_name in ("proc3", "proc4"):
                                note.write(
                                    "2) Band-pass: FFT mask 40-80 MHz on each trace.\n"
                                )
                                note.write(
                                    f"3) 2-D mean filter (XxZ): {args.mean_size_post}x"
                                    f"{args.mean_size_post} boxcar window.\n"
                                )
                                if proc_name == "proc4":
                                    note.write(
                                        f"4) 2-D mean filter (YxZ): {args.mean_size_pre}x"
                                        f"{args.mean_size_pre} boxcar window.\n"
                                    )
                                    note.write("5) power dB max: Hilbert magnitude.\n")
                                else:
                                    note.write("4) power dB max: Hilbert magnitude.\n")
                            else:
                                note.write(
                                    f"2) 2-D mean filter (XxZ): {args.mean_size_post}x"
                                    f"{args.mean_size_post} boxcar window.\n"
                                )
                                if proc_name == "proc1":
                                    note.write(
                                        f"3) 2-D mean filter (YxZ): {args.mean_size_pre}x"
                                        f"{args.mean_size_pre} boxcar window.\n"
                                    )
                                    note.write("4) power dB max: Hilbert magnitude.\n")
                                else:
                                    note.write("3) power dB max: Hilbert magnitude.\n")
                        else:
                            note.write(
                                f"2) 2-D mean filter (XxZ): {args.mean_size_post}x"
                                f"{args.mean_size_post} boxcar window.\n"
                            )
                            note.write(
                                f"3) 2-D mean filter (YxZ): {args.mean_size_pre}x"
                                f"{args.mean_size_pre} boxcar window.\n"
                            )
                            note.write("4) power dB max: Hilbert magnitude.\n")
                        time_zero_step = 6 if proc_name == "procedure_1" else 5
                        if proc_name == "no_mig_no_bandpass":
                            time_zero_step = 5
                        elif proc_name in new_proc_names:
                            if proc_name == "proc1":
                                time_zero_step = 5
                            elif proc_name == "proc2":
                                time_zero_step = 4
                            elif proc_name == "proc3":
                                time_zero_step = 5
                            elif proc_name == "proc4":
                                time_zero_step = 6
                        elif proc_name in new_mig_names:
                            if proc_name == "proc1m":
                                time_zero_step = 6
                            elif proc_name == "proc2m":
                                time_zero_step = 5
                            elif proc_name == "proc3m":
                                time_zero_step = 6
                            elif proc_name == "proc4m":
                                time_zero_step = 7
                        note.write(
                            f"{time_zero_step}) (none) Time-zero correction: "
                            "not applied per request.\n"
                        )
                        note.write("\nPlotting:\n")
                        if args.agc_plots:
                            note.write("AGC applied for PNGs.\n")
                        else:
                            note.write("AGC not applied for PNGs.\n")
                        note.write("Zoom range for plots: samples 500-1500.\n")
                    note_written.add(proc_name)

            if (
                not args.no_plots
                and "procedure_1" in final_steps
                and "procedure_2" in final_steps
            ):
                compare_full_dir = os.path.join(
                    full_root, "compare", f"line_{line_id:02d}"
                )
                compare_zoom_dir = os.path.join(
                    zoom_root, "compare", f"line_{line_id:02d}"
                )
                os.makedirs(compare_full_dir, exist_ok=True)
                os.makedirs(compare_zoom_dir, exist_ok=True)
                for agc_ns in agc_windows_ns:
                    save_compare_radargrams(
                        final_steps["procedure_1"],
                        final_steps["procedure_2"],
                        dt,
                        5,
                        "after Kirchhoff migration (P1 vs P2)",
                        zoom_range,
                        compare_full_dir,
                        compare_zoom_dir,
                        agc_ns,
                        trace_spacing,
                        max_depth,
                        label_a="Procedure 1",
                        label_b="Procedure 2",
                    )
    finally:
        for handle_map in output_handles.values():
            for handle in handle_map.values():
                handle.close()


if __name__ == "__main__":
    main()
