
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, radon
from skimage.filters import sobel
from scipy.ndimage import gaussian_filter


DEFAULT_CSV_PATH = "./Section 3 data.csv"
OUT_DIR = "./results"

DEPTH_RES_M = 1.0
TIME_RES_MIN = 1.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def robust_normalize(X: np.ndarray, clip_p: Tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    lo, hi = np.percentile(X, clip_p)
    if hi <= lo:
        return np.zeros_like(X, dtype=np.float32)
    Xc = np.clip(X, lo, hi)
    return ((Xc - lo) / (hi - lo)).astype(np.float32)


def zscore_along_axis(X: np.ndarray, axis: int) -> np.ndarray:
    mu = X.mean(axis=axis, keepdims=True)
    sd = X.std(axis=axis, keepdims=True) + 1e-9
    return (X - mu) / sd


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@dataclass
class LineSegment:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def dx(self) -> float:
        return float(self.x2 - self.x1)

    @property
    def dy(self) -> float:
        return float(self.y2 - self.y1)

    @property
    def length_px(self) -> float:
        return float(math.hypot(self.dx, self.dy))

    def slope_depth_per_time(self) -> Optional[float]:
        if abs(self.dx) < 1e-9:
            return None
        return self.dy / self.dx


def segment_velocity_m_per_min(seg: LineSegment) -> Optional[float]:
    s = seg.slope_depth_per_time()
    if s is None:
        return None
    return (s * DEPTH_RES_M) / TIME_RES_MIN


def segment_spans(seg: LineSegment) -> Tuple[float, float]:
    depth_span_m = abs(seg.dy) * DEPTH_RES_M
    duration_min = abs(seg.dx) * TIME_RES_MIN
    return depth_span_m, duration_min


def load_das_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    X = df.values.astype(np.float32)
    return X


def eda_plots(X: np.ndarray, out_dir: str) -> None:
    ensure_dir(out_dir)

    flat = X.ravel()
    stats = {
        "shape": X.shape,
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "p1": float(np.percentile(flat, 1)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(flat.max()),
    }
    print("\n=== DATA SUMMARY ===")
    for k, v in stats.items():
        print(f"{k:>6}: {v}")

    plt.figure(figsize=(11, 6))
    plt.imshow(X, aspect="auto", origin="upper")
    plt.title("DAS spatiotemporal map (raw)")
    plt.xlabel("Time index (minute)")
    plt.ylabel("Depth index (meter)")
    plt.colorbar(label="Amplitude")
    save_fig(os.path.join(out_dir, "heatmap_raw.png"))

    Xn = robust_normalize(X, (1, 99))
    plt.figure(figsize=(11, 6))
    plt.imshow(Xn, aspect="auto", origin="upper")
    plt.title("DAS spatiotemporal map (robust normalized p1–p99)")
    plt.xlabel("Time index (minute)")
    plt.ylabel("Depth index (meter)")
    plt.colorbar(label="Normalized amplitude")
    save_fig(os.path.join(out_dir, "heatmap_norm.png"))

    plt.figure(figsize=(8, 5))
    plt.hist(flat, bins=80)
    plt.title("Amplitude histogram (raw)")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    save_fig(os.path.join(out_dir, "hist_raw.png"))

    mean_t = X.mean(axis=0)
    std_t = X.std(axis=0)
    t = np.arange(X.shape[1]) * TIME_RES_MIN

    plt.figure(figsize=(10, 4))
    plt.plot(t, mean_t)
    plt.title("Mean amplitude over time")
    plt.xlabel("Time (min)")
    plt.ylabel("Mean amplitude")
    save_fig(os.path.join(out_dir, "mean_over_time.png"))

    plt.figure(figsize=(10, 4))
    plt.plot(t, std_t)
    plt.title("Std amplitude over time")
    plt.xlabel("Time (min)")
    plt.ylabel("Std amplitude")
    save_fig(os.path.join(out_dir, "std_over_time.png"))

    mean_d = X.mean(axis=1)
    std_d = X.std(axis=1)
    d = np.arange(X.shape[0]) * DEPTH_RES_M

    plt.figure(figsize=(10, 4))
    plt.plot(mean_d, d)
    plt.gca().invert_yaxis()
    plt.title("Mean amplitude over depth")
    plt.xlabel("Mean amplitude")
    plt.ylabel("Depth (m)")
    save_fig(os.path.join(out_dir, "mean_over_depth.png"))

    plt.figure(figsize=(10, 4))
    plt.plot(std_d, d)
    plt.gca().invert_yaxis()
    plt.title("Std amplitude over depth")
    plt.xlabel("Std amplitude")
    plt.ylabel("Depth (m)")
    save_fig(os.path.join(out_dir, "std_over_depth.png"))


def detect_lines_hough(
    X: np.ndarray,
    out_dir: str,
    sigma_smooth: float = 1.0,
    canny_sigma: float = 1.4,
    canny_low: float = 0.10,
    canny_high: float = 0.30,
    line_length: int = 40,
    line_gap: int = 5,
) -> List[LineSegment]:
    ensure_dir(out_dir)

    Xn = robust_normalize(X, (1, 99))
    Xs = gaussian_filter(Xn, sigma=sigma_smooth) if sigma_smooth > 0 else Xn

    edges = canny(Xs, sigma=canny_sigma, low_threshold=canny_low, high_threshold=canny_high)

    lines = probabilistic_hough_line(
        edges,
        threshold=10,
        line_length=line_length,
        line_gap=line_gap,
    )

    segs: List[LineSegment] = []
    for (x1, y1), (x2, y2) in lines:
        segs.append(LineSegment(int(x1), int(y1), int(x2), int(y2)))

    plt.figure(figsize=(11, 6))
    plt.imshow(edges, aspect="auto", origin="upper")
    plt.title("Canny edge map (for Hough)")
    plt.xlabel("Time index (minute)")
    plt.ylabel("Depth index (meter)")
    save_fig(os.path.join(out_dir, "edges_canny.png"))

    plt.figure(figsize=(11, 6))
    plt.imshow(Xn, aspect="auto", origin="upper")
    for s in segs:
        plt.plot([s.x1, s.x2], [s.y1, s.y2], linewidth=2)
    plt.title(f"Hough line segments overlay (n={len(segs)})")
    plt.xlabel("Time index (minute)")
    plt.ylabel("Depth index (meter)")
    save_fig(os.path.join(out_dir, "lines_hough_overlay.png"))

    return segs


def dominant_angle_radon(
    X: np.ndarray,
    out_dir: str,
    theta: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    ensure_dir(out_dir)

    Xn = robust_normalize(X, (1, 99))
    Xg = sobel(Xn)

    if theta is None:
        theta = np.linspace(0.0, 180.0, 181)

    R = radon(Xg, theta=theta, circle=False)

    energy = (R ** 2).sum(axis=0)
    best_idx = int(np.argmax(energy))
    best_theta = float(theta[best_idx])

    plt.figure(figsize=(9, 4))
    plt.plot(theta, energy)
    plt.title("Radon energy vs angle (dominant line orientation)")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Energy")
    save_fig(os.path.join(out_dir, "radon_energy.png"))

    return best_theta, theta, energy


def summarize_segments(segs: List[LineSegment], top_k: int = 15) -> pd.DataFrame:
    rows = []
    for s in segs:
        v = segment_velocity_m_per_min(s)
        depth_span_m, duration_min = segment_spans(s)
        rows.append(
            {
                "x1_time": s.x1,
                "y1_depth": s.y1,
                "x2_time": s.x2,
                "y2_depth": s.y2,
                "length_px": s.length_px,
                "slope_d_per_t": s.slope_depth_per_time(),
                "velocity_m_per_min": v,
                "velocity_m_per_s": (v / 60.0) if v is not None else None,
                "depth_span_m": depth_span_m,
                "duration_min": duration_min,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("length_px", ascending=False).head(top_k).reset_index(drop=True)
    return df


def main(csv_path: str = DEFAULT_CSV_PATH, out_dir: str = OUT_DIR) -> None:
    ensure_dir(out_dir)

    X = load_das_csv(csv_path)
    print(f"Loaded: {csv_path}")
    print(f"Matrix shape: {X.shape[0]} depths x {X.shape[1]} times")

    eda_plots(X, out_dir)

    best_theta, theta_grid, energy = dominant_angle_radon(X, out_dir)
    print(f"\nDominant Radon angle (degrees): {best_theta:.1f}")

    segs = detect_lines_hough(
        X,
        out_dir,
        sigma_smooth=1.0,
        canny_sigma=1.4,
        canny_low=0.10,
        canny_high=0.30,
        line_length=40,
        line_gap=5,
    )
    print(f"\nDetected {len(segs)} line segments via Hough.")

    df = summarize_segments(segs, top_k=20)
    if df.empty:
        print("No segments found.")
    else:
        print("\n=== TOP LINE SEGMENTS (by pixel length) ===")
        with pd.option_context("display.max_columns", 20, "display.width", 160):
            print(df)

        out_csv = os.path.join(out_dir, "line_segments_top.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nSaved segment summary to: {out_csv}")

    print(f"\nAll figures saved in: {out_dir}")


if __name__ == "__main__":
    main()