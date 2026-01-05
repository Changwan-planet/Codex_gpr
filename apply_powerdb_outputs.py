#!/usr/bin/env python3
import argparse
import os
import numpy as np


def analytic_magnitude(data):
    n_samples = data.shape[0]
    spectrum = np.fft.fft(data, axis=0)
    phase_shift = np.empty_like(spectrum)
    pos_mask = np.arange(n_samples) <= (n_samples // 2 - 1)
    phase_shift[pos_mask, :] = spectrum[pos_mask, :] * np.exp(-1j * np.pi / 2.0)
    phase_shift[~pos_mask, :] = spectrum[~pos_mask, :] * np.exp(1j * np.pi / 2.0)
    hilbert = np.fft.ifft(phase_shift, axis=0).real
    return np.sqrt(data**2 + hilbert**2)


def load_cube_from_txt(path):
    max_trace = 0
    max_sample = 0
    line_ids = set()
    with open(path, "r", encoding="ascii", errors="ignore") as handle:
        for row in handle:
            parts = row.split()
            if len(parts) != 4:
                continue
            trace_idx = int(parts[0])
            line_idx = int(parts[1])
            sample_idx = int(parts[2])
            line_ids.add(line_idx)
            if trace_idx > max_trace:
                max_trace = trace_idx
            if sample_idx > max_sample:
                max_sample = sample_idx

    if max_trace == 0 or max_sample == 0 or not line_ids:
        raise RuntimeError("No data found in input file.")

    line_ids_sorted = sorted(line_ids)
    line_index = {line_id: idx for idx, line_id in enumerate(line_ids_sorted)}
    data = np.zeros((max_sample, max_trace, len(line_ids_sorted)), dtype=np.float64)

    with open(path, "r", encoding="ascii", errors="ignore") as handle:
        for row in handle:
            parts = row.split()
            if len(parts) != 4:
                continue
            trace_idx = int(parts[0])
            line_idx = int(parts[1])
            sample_idx = int(parts[2])
            data[sample_idx - 1, trace_idx - 1, line_index[line_idx]] = float(
                parts[3]
            )

    return data, line_ids_sorted


def write_cube(path, cube, line_ids):
    n_samples, n_traces, _ = cube.shape
    with open(path, "w", encoding="ascii") as handle:
        for line_id, line_idx in line_ids:
            for trace_idx in range(n_traces):
                for sample_idx in range(n_samples):
                    handle.write(
                        f"{trace_idx + 1} {line_id} {sample_idx + 1} "
                        f"{cube[sample_idx, trace_idx, line_idx]:.6f}\n"
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="3DCUBE_GPR_BA5_new2Df.txt",
        help="Input amplitude file.",
    )
    args = parser.parse_args()

    data, line_ids = load_cube_from_txt(args.input)
    line_pairs = [(line_id, idx) for idx, line_id in enumerate(line_ids)]

    power = analytic_magnitude(data)
    max_per_trace = np.max(power, axis=0)
    max_per_trace[max_per_trace == 0.0] = 1.0
    power_db_max = 10.0 * np.log10((power**2) / (max_per_trace**2))

    slice_mean = np.mean(power, axis=(1, 2))
    slice_mean[slice_mean == 0.0] = 1.0
    power_db_mean = 10.0 * np.log10((power**2) / (slice_mean[:, None, None] ** 2))

    base = os.path.splitext(args.input)[0]
    out_max = f"{base}_pdBmax.txt"
    out_mean = f"{base}_pdBmean.txt"

    write_cube(out_max, power_db_max, line_pairs)
    write_cube(out_mean, power_db_mean, line_pairs)

    print(f"Wrote {out_max}")
    print(f"Wrote {out_mean}")


if __name__ == "__main__":
    main()
