import math
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from plugins.base_plugin import FilterPlugin


def ensure_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        return image
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sample_gray_at_points(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    xs_clipped = np.clip(xs, 0, w - 1.001)
    ys_clipped = np.clip(ys, 0, h - 1.001)

    x0 = np.floor(xs_clipped).astype(np.int32)
    y0 = np.floor(ys_clipped).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = xs_clipped - x0
    dy = ys_clipped - y0

    Ia = gray[y0, x0].astype(np.float32)
    Ib = gray[y0, x1].astype(np.float32)
    Ic = gray[y1, x0].astype(np.float32)
    Id = gray[y1, x1].astype(np.float32)

    wa = (1.0 - dx) * (1.0 - dy)
    wb = dx * (1.0 - dy)
    wc = (1.0 - dx) * dy
    wd = dx * dy

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def sample_profile_along_line(
    gray: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    step_px: Optional[float] = None,
    num_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0 = p0
    x1, y1 = p1
    vx = x1 - x0
    vy = y1 - y0
    L = math.hypot(vx, vy)
    if L <= 0:
        return np.array([]), np.array([]), np.empty((0, 2), dtype=np.float32)

    if step_px is not None and step_px > 0:
        n = max(2, int(math.ceil(L / float(step_px))))
    elif num_samples is not None and num_samples > 1:
        n = int(num_samples)
    else:
        n = max(2, int(L))

    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    xs = x0 + vx * t
    ys = y0 + vy * t
    s = t * L
    I = sample_gray_at_points(gray, xs, ys)
    pts = np.stack([xs, ys], axis=1)
    return s, I, pts


def smooth_profile(profile: np.ndarray, mode: str, gaussian_sigma: float, sg_window: int, sg_poly: int) -> np.ndarray:
    if profile.size == 0:
        return profile
    if mode == "Gaussian" and gaussian_sigma > 0:
        # approximate kernel size from sigma (odd)
        ksize = int(max(3, 6 * gaussian_sigma))
        if ksize % 2 == 0:
            ksize += 1
        p = profile.astype(np.float32).reshape(1, -1)
        sm = cv2.GaussianBlur(p, (ksize, 1), gaussian_sigma, borderType=cv2.BORDER_REPLICATE)
        return sm.reshape(-1)
    if mode == "Savitzky-Golay":
        window = sg_window if sg_window % 2 == 1 and sg_window >= 3 else 5
        poly = max(1, min(sg_poly, window - 1))
        half = window // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        A = np.vander(x, poly + 1, increasing=True)
        ATA = A.T @ A
        try:
            ATA_inv = np.linalg.inv(ATA)
        except np.linalg.LinAlgError:
            return profile
        coeffs = ATA_inv @ A.T
        # smoothing corresponds to evaluating polynomial at 0 -> first row (index 0)
        kernel = coeffs[0]
        pad_left = np.repeat(profile[:1], half)
        pad_right = np.repeat(profile[-1:], half)
        padded = np.concatenate([pad_left, profile, pad_right]).astype(np.float64)
        sm = np.convolve(padded, kernel[::-1], mode="valid")
        return sm.astype(np.float32)
    return profile


def normalize_0_1(profile: np.ndarray) -> np.ndarray:
    if profile.size == 0:
        return profile
    p_min = float(profile.min())
    p_max = float(profile.max())
    if p_max <= p_min:
        return np.zeros_like(profile, dtype=np.float32)
    return ((profile - p_min) / (p_max - p_min)).astype(np.float32)


def compute_derivative_and_metrics(
    s: np.ndarray,
    I: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, int]]:
    n = len(I)
    if n < 3:
        return np.zeros_like(I, dtype=np.float32), {}, {}
    ds = float(np.mean(np.diff(s))) if n > 1 else 1.0
    dI = np.zeros_like(I, dtype=np.float32)
    # central differences
    dI[1:-1] = (I[2:] - I[:-2]) / (2.0 * ds)
    dI[0] = (I[1] - I[0]) / ds
    dI[-1] = (I[-1] - I[-2]) / ds

    peak_height_pos = float(np.max(dI))
    peak_height_neg = float(abs(np.min(dI)))
    peak_to_peak = peak_height_pos + peak_height_neg

    pos_part = np.maximum(dI, 0.0)
    neg_part = np.maximum(-dI, 0.0)
    area_pos = float(np.sum(pos_part * ds))
    area_neg = float(np.sum(neg_part * ds))
    area_total = area_pos + area_neg

    pos_idx = int(np.argmax(dI))
    neg_idx = int(np.argmin(dI))

    metrics = {
        "peak_height_pos": peak_height_pos,
        "peak_height_neg": peak_height_neg,
        "peak_to_peak": peak_to_peak,
        "area_pos": area_pos,
        "area_neg": area_neg,
        "area_total": area_total,
    }
    peak_indices = {"pos_index": pos_idx, "neg_index": neg_idx}
    return dI, metrics, peak_indices


def estimate_edge_width_10_90(s: np.ndarray, I_norm: np.ndarray) -> Tuple[Optional[float], Optional[List[int]]]:
    if I_norm.size < 3:
        return None, None
    # Decide rising or falling edge from endpoints
    start = float(I_norm[0])
    end = float(I_norm[-1])
    rising = end >= start
    prof = I_norm if rising else (1.0 - I_norm)

    def find_cross(x, level):
        vals = prof - level
        sign = np.sign(vals)
        for i in range(len(vals) - 1):
            if sign[i] == 0:
                return s[i], i
            if sign[i] < 0 and sign[i + 1] > 0:
                # linear interpolation
                t = -vals[i] / (vals[i + 1] - vals[i] + 1e-12)
                ss = s[i] + t * (s[i + 1] - s[i])
                idx = i if t < 0.5 else i + 1
                return ss, idx
        return None, None

    s10, i10 = find_cross(s, 0.1)
    s90, i90 = find_cross(s, 0.9)
    if s10 is None or s90 is None:
        return None, None
    return float(abs(s90 - s10)), [i10, i90]


def discretize_evaluation_path(p0: Tuple[float, float], p1: Tuple[float, float], M: int) -> List[Tuple[float, float]]:
    if M <= 1:
        return [((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)]
    xs = np.linspace(p0[0], p1[0], M, dtype=np.float32)
    ys = np.linspace(p0[1], p1[1], M, dtype=np.float32)
    return list(zip(xs.tolist(), ys.tolist()))


def compute_gradients(gray: np.ndarray, method: str, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    img = gray
    if sigma > 0:
        ksize = int(max(3, 6 * sigma))
        if ksize % 2 == 0:
            ksize += 1
        img = cv2.GaussianBlur(gray, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE)
    if method == "Scharr":
        gx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    else:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def sample_gradient_at(
    gx: np.ndarray,
    gy: np.ndarray,
    center: Tuple[float, float],
    radius: int,
) -> Tuple[float, float]:
    cx, cy = center
    if radius <= 0:
        val_x = sample_gray_at_points(gx, np.array([cx]), np.array([cy]))[0]
        val_y = sample_gray_at_points(gy, np.array([cx]), np.array([cy]))[0]
        return float(val_x), float(val_y)

    xs = []
    ys = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xs.append(cx + dx)
            ys.append(cy + dy)
    xs_arr = np.asarray(xs, dtype=np.float32)
    ys_arr = np.asarray(ys, dtype=np.float32)
    vals_x = sample_gray_at_points(gx, xs_arr, ys_arr)
    vals_y = sample_gray_at_points(gy, xs_arr, ys_arr)
    return float(np.mean(vals_x)), float(np.mean(vals_y))


def sample_cross_section(
    gray: np.ndarray,
    center: Tuple[float, float],
    direction: Tuple[float, float],
    length_W: float,
    samples_per_section: int,
    multi_line_K: int = 1,
    offset_d: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if samples_per_section < 2 or length_W <= 0:
        return np.array([]), np.array([]), np.empty((0, 2)), np.empty((0, 2)), np.array([])

    cx, cy = center
    dx, dy = direction
    norm = math.hypot(dx, dy)
    if norm == 0:
        return np.array([]), np.array([]), np.empty((0, 2)), np.empty((0, 2)), np.array([])
    dx /= norm
    dy /= norm

    half = length_W * 0.5
    u = np.linspace(-half, half, samples_per_section, dtype=np.float32)

    # Secondary normal for multi-line offsets
    mx, my = -dy, dx

    profiles = []

    def sample_single(offset: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = cx + u * dx + offset * mx
        ys = cy + u * dy + offset * my
        I = sample_gray_at_points(gray, xs, ys)
        pts = np.stack([xs, ys], axis=1)
        return u.copy(), I, pts

    if multi_line_K is None or multi_line_K <= 1 or offset_d == 0.0:
        u_coords, I, pts = sample_single(0.0)
        return u_coords, I, pts, np.array([pts[0], pts[-1]]), np.full_like(I, fill_value=1.0, dtype=np.float32)

    k_vals = np.arange(-(multi_line_K - 1) / 2.0, (multi_line_K - 1) / 2.0 + 1.0, 1.0, dtype=np.float32)
    all_profiles = []
    for kv in k_vals:
        _, I_k, pts_k = sample_single(float(kv * offset_d))
        all_profiles.append(I_k)
    all_profiles_arr = np.stack(all_profiles, axis=0)  # K x N
    I_mean = np.mean(all_profiles_arr, axis=0)
    u_coords, _, pts0 = sample_single(0.0)
    endpoints = np.array([pts0[0], pts0[-1]], dtype=np.float32)
    weights = np.ones_like(I_mean, dtype=np.float32)
    return u_coords, I_mean, pts0, endpoints, weights


class EdgeMeasurement(FilterPlugin):
    """Edge Measurement tool: along-line and cross-section modes with gradient alignment."""

    def __init__(self):
        super().__init__()
        self.name = "Edge Measurement"
        self.description = "Measure edge sharpness along a manual line or cross-sections (with optional gradient alignment)."
        self.parameters = {
            # Mode selection
            "Measurement mode": ["Along line (single profile)", "Perpendicular cross-sections (multi profile)"],
            # Sampling along main line
            "Sampling mode": ["Step (px)", "Num samples"],
            "Step (px)": 0.5,
            "Num samples": 500,
            # Cross-sections
            "Cross-section count M": 21,
            "Cross-section length W (px)": 20.0,
            "Samples per cross-section": 64,
            # Gradient alignment
            "Align cross-sections to local gradient": False,
            "Gradient method": ["Sobel", "Scharr"],
            "Gradient sigma": 1.0,
            "Min gradient magnitude": 5.0,
            "Gradient radius r": 1,
            # Robustness / multi-line averaging
            "Multi-line average enabled": False,
            "Multi-line count K": 3,
            "Multi-line offset d (px)": 0.5,
            # Smoothing and normalization
            "Smoothing type": ["None", "Gaussian", "Savitzky-Golay"],
            "Gaussian sigma": 1.0,
            "SavGol window": 7,
            "SavGol polyorder": 2,
            "Normalize 0-1": True,
            # Visualization
            "Cross-section overlay stride": 3,
            "Overlay metric": ["peak_to_peak", "edge_width"],
        }
        self.image_viewer = None
        self.results_panel = None

    def set_context(self, image_viewer, edge_panel=None):
        self.image_viewer = image_viewer
        self.results_panel = edge_panel

    def _get_current_line(self) -> Optional[Tuple[float, float, float, float]]:
        if self.image_viewer is None:
            return None
        if not self.image_viewer.measurements:
            return None
        x1, y1, x2, y2, _, _, _ = self.image_viewer.measurements[-1]
        return float(x1), float(y1), float(x2), float(y2)

    def _build_params_summary(self) -> str:
        p = self.parameters
        # Keep this human-readable and multi-line for the UI.
        lines = [
            f"mode: {p.get('Measurement mode')}",
            f"sampling: {p.get('Sampling mode')}  step={p.get('Step (px)')}  N={p.get('Num samples')}",
            f"cross-sections: M={p.get('Cross-section count M')}  W={p.get('Cross-section length W (px)')}  Ns={p.get('Samples per cross-section')}",
            f"smoothing: {p.get('Smoothing type')}  sigma={p.get('Gaussian sigma')}  sg_window={p.get('SavGol window')}  sg_poly={p.get('SavGol polyorder')}",
            f"normalize: {p.get('Normalize 0-1')}",
            f"gradient: align={p.get('Align cross-sections to local gradient')}  method={p.get('Gradient method')}  "
            f"sigma={p.get('Gradient sigma')}  min_mag={p.get('Min gradient magnitude')}  r={p.get('Gradient radius r')}",
            f"multi-line: enabled={p.get('Multi-line average enabled')}  K={p.get('Multi-line count K')}  d={p.get('Multi-line offset d (px)')}",
        ]
        return "\n".join(str(x) for x in lines)

    def process(self, image):
        # Clear overlays/results if context is missing
        if self.image_viewer is None or self.results_panel is None:
            return image

        # Some parameters are stored as lists initially to populate combo boxes.
        # This helper converts them to concrete string choices for internal use.
        def _choice(val, default=None):
            if isinstance(val, (list, tuple)):
                return val[0] if val else default
            return val if val is not None else default

        line = self._get_current_line()
        if line is None:
            self.image_viewer.clear_edge_overlay()
            self.results_panel.set_results(
                {
                    "mode": "No line",
                    "params_summary": "No measurement line defined. Use the Measure tool to draw a line.",
                    "single_metrics": {},
                    "aggregate_metrics": {},
                    "cross_sections": [],
                    "profile": None,
                    "image_name": "",
                }
            )
            return image

        x1, y1, x2, y2 = line

        # Map into ROI-local coordinates
        roi = self.image_viewer.get_roi()
        if isinstance(roi, tuple) and len(roi) == 4:
            rx, ry, rw, rh = roi
            # image passed to us is roi-image, so subtract roi origin
            p0 = (x1 - rx, y1 - ry)
            p1 = (x2 - rx, y2 - ry)
            # For overlays we still need full-image coordinates
            roi_offset = (rx, ry)
        else:
            p0 = (x1, y1)
            p1 = (x2, y2)
            roi_offset = (0.0, 0.0)

        gray = ensure_gray(image)
        if gray is None:
            return image

        mode = _choice(self.parameters.get("Measurement mode"), "Along line (single profile)")
        params_summary = self._build_params_summary()

        # Common smoothing/normalization params
        smoothing_type = _choice(self.parameters.get("Smoothing type"), "None")
        gaussian_sigma = float(self.parameters.get("Gaussian sigma"))
        sg_window = int(self.parameters.get("SavGol window"))
        sg_poly = int(self.parameters.get("SavGol polyorder"))
        do_norm = bool(self.parameters.get("Normalize 0-1"))

        image_name = getattr(self.image_viewer, "image_path", "") if hasattr(self.image_viewer, "image_path") else ""

        if isinstance(mode, str) and mode.startswith("Along"):
            # Mode A: single profile along the line
            sampling_mode = _choice(self.parameters.get("Sampling mode"), "Step (px)")
            if sampling_mode == "Step (px)":
                step_px = float(self.parameters.get("Step (px)"))
                num_samples = None
            else:
                step_px = None
                num_samples = int(self.parameters.get("Num samples"))

            s, I_raw, pts = sample_profile_along_line(gray, p0, p1, step_px=step_px, num_samples=num_samples)
            if s.size == 0:
                return image

            I = I_raw.astype(np.float32)
            I_sm = smooth_profile(I, smoothing_type, gaussian_sigma, sg_window, sg_poly)
            I_proc = normalize_0_1(I_sm) if do_norm else I_sm
            dI, metrics, peak_indices = compute_derivative_and_metrics(s, I_proc)
            width_val, width_indices = estimate_edge_width_10_90(s, normalize_0_1(I_sm))
            metrics["edge_width"] = width_val if width_val is not None else float("nan")

            # Map peaks and width markers back to full-image coordinates
            peaks_xy = []
            if "pos_index" in peak_indices:
                idx = peak_indices["pos_index"]
                if 0 <= idx < pts.shape[0]:
                    px, py = pts[idx]
                    peaks_xy.append((px + roi_offset[0], py + roi_offset[1]))
            if "neg_index" in peak_indices:
                idx = peak_indices["neg_index"]
                if 0 <= idx < pts.shape[0]:
                    px, py = pts[idx]
                    peaks_xy.append((px + roi_offset[0], py + roi_offset[1]))

            width_points = []
            if width_indices:
                for wi in width_indices:
                    if 0 <= wi < pts.shape[0]:
                        px, py = pts[wi]
                        width_points.append((px + roi_offset[0], py + roi_offset[1]))

            overlay = {
                "peaks": peaks_xy,
                "width_points": width_points,
                "cross_sections": [],
                "heat_points": [],
            }
            self.image_viewer.set_edge_overlay(overlay)

            profile_dict = {
                "s": s.tolist(),
                "I": I_proc.tolist(),
                "dI": dI.tolist(),
                "peaks": peak_indices,
                "width_points": {"indices": width_indices} if width_indices else {},
            }
            results = {
                "mode": mode,
                "params_summary": params_summary,
                "single_metrics": metrics,
                "aggregate_metrics": {},
                "cross_sections": [],
                "profile": profile_dict,
                "image_name": image_name,
            }
            self.results_panel.set_results(results)
            return image

        # Mode B/C: cross-sections
        M = int(self.parameters.get("Cross-section count M"))
        W = float(self.parameters.get("Cross-section length W (px)"))
        Ns = int(self.parameters.get("Samples per cross-section"))
        align_grad = bool(self.parameters.get("Align cross-sections to local gradient"))
        grad_method = _choice(self.parameters.get("Gradient method"), "Sobel")
        grad_sigma = float(self.parameters.get("Gradient sigma"))
        grad_min = float(self.parameters.get("Min gradient magnitude"))
        grad_radius = int(self.parameters.get("Gradient radius r"))
        multi_enabled = bool(self.parameters.get("Multi-line average enabled"))
        multi_K = int(self.parameters.get("Multi-line count K"))
        multi_d = float(self.parameters.get("Multi-line offset d (px)"))
        overlay_metric_name = _choice(self.parameters.get("Overlay metric"), "peak_to_peak")
        stride = max(1, int(self.parameters.get("Cross-section overlay stride")))

        centers = discretize_evaluation_path(p0, p1, M)
        vx = p1[0] - p0[0]
        vy = p1[1] - p0[1]
        vnorm = math.hypot(vx, vy)
        if vnorm == 0:
            return image
        tx = vx / vnorm
        ty = vy / vnorm
        # Default cross-section direction is perpendicular to evaluation path
        nx_default = -ty
        ny_default = tx

        gx = gy = None
        if align_grad:
            gx, gy = compute_gradients(gray, grad_method if isinstance(grad_method, str) else "Sobel", grad_sigma)

        cs_results = []
        cs_lines_for_overlay = []
        heat_points = []

        for idx, c in enumerate(centers):
            cx, cy = c
            # Determine direction
            nx, ny = nx_default, ny_default
            if align_grad and gx is not None and gy is not None:
                gx_val, gy_val = sample_gradient_at(gx, gy, (cx, cy), grad_radius)
                mag = math.hypot(gx_val, gy_val)
                if mag >= grad_min:
                    nx = gx_val / mag
                    ny = gy_val / mag

            if multi_enabled:
                K = max(1, multi_K)
                d_off = float(multi_d)
            else:
                K = 1
                d_off = 0.0

            u_coords, I_raw, pts, endpoints_local, weights = sample_cross_section(
                gray,
                (cx, cy),
                (nx, ny),
                W,
                Ns,
                multi_line_K=K,
                offset_d=d_off,
            )
            if I_raw.size == 0:
                continue

            I_sm = smooth_profile(I_raw.astype(np.float32), smoothing_type, gaussian_sigma, sg_window, sg_poly)
            I_proc = normalize_0_1(I_sm) if do_norm else I_sm
            s_cs = u_coords.copy()
            dI_cs, metrics_cs, peak_indices_cs = compute_derivative_and_metrics(s_cs, I_proc)
            width_val_cs, width_idx_cs = estimate_edge_width_10_90(s_cs, normalize_0_1(I_sm))
            metrics_cs["edge_width"] = width_val_cs if width_val_cs is not None else float("nan")

            # Geometry in full-image coordinates
            pts_full = pts + np.array(roi_offset, dtype=np.float32)
            center_full = (pts_full[Ns // 2][0], pts_full[Ns // 2][1])
            endpoints_full = endpoints_local + np.array(roi_offset, dtype=np.float32)
            p0_full = (endpoints_full[0][0], endpoints_full[0][1])
            p1_full = (endpoints_full[1][0], endpoints_full[1][1])
            direction_full = (nx, ny)

            if idx % stride == 0:
                cs_lines_for_overlay.append((p0_full[0], p0_full[1], p1_full[0], p1_full[1]))

            # Heat metric (normalized later)
            metric_for_heat = metrics_cs.get(overlay_metric_name, metrics_cs.get("peak_to_peak"))
            heat_points.append((center_full[0], center_full[1], metric_for_heat))

            profile_dict = {
                "s": s_cs.tolist(),
                "I": I_proc.tolist(),
                "dI": dI_cs.tolist(),
                "peaks": peak_indices_cs,
                "width_points": {"indices": width_idx_cs} if width_idx_cs else {},
            }

            cs_results.append(
                {
                    "index": idx,
                    "geometry": {
                        "center": center_full,
                        "p0": p0_full,
                        "p1": p1_full,
                        "direction": direction_full,
                    },
                    "metrics": metrics_cs,
                    "profile": profile_dict,
                }
            )

        if not cs_results:
            self.image_viewer.clear_edge_overlay()
            self.results_panel.set_results(
                {
                    "mode": mode,
                    "params_summary": params_summary,
                    "single_metrics": {},
                    "aggregate_metrics": {},
                    "cross_sections": [],
                    "profile": None,
                    "image_name": image_name,
                }
            )
            return image

        # Aggregate metrics
        metric_names = ["peak_height_pos", "peak_height_neg", "peak_to_peak", "area_pos", "area_neg", "area_total", "edge_width"]
        agg = {}
        for name in metric_names:
            vals = np.array(
                [cs["metrics"].get(name) for cs in cs_results if cs["metrics"].get(name) is not None],
                dtype=np.float32,
            )
            if vals.size == 0:
                continue
            agg[name] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90)),
            }
        agg["cross_section_count"] = len(cs_results)
        agg["valid_cross_sections"] = len(cs_results)

        # Best/worst indices by chosen metric
        metric_array = np.array(
            [cs["metrics"].get(overlay_metric_name, cs["metrics"].get("peak_to_peak")) for cs in cs_results],
            dtype=np.float32,
        )
        best_idx = int(np.argmax(metric_array))
        worst_idx = int(np.argmin(metric_array))

        # Normalize heat strengths to [0,1]
        strengths = metric_array.copy()
        m_min = float(np.min(strengths))
        m_max = float(np.max(strengths))
        heat_norm = []
        if m_max > m_min:
            for (x, y, _), val in zip(heat_points, strengths):
                heat_norm.append((x, y, (val - m_min) / (m_max - m_min)))
        else:
            for (x, y, _), _ in zip(heat_points, strengths):
                heat_norm.append((x, y, 0.5))

        overlay = {
            "peaks": [],  # peaks are visualized via selected cross-section profile
            "width_points": [],
            "cross_sections": cs_lines_for_overlay,
            "heat_points": heat_norm,
        }
        self.image_viewer.set_edge_overlay(overlay)

        # Default profile to display: best cross-section
        default_cs = cs_results[best_idx]

        results = {
            "mode": mode,
            "params_summary": params_summary,
            "single_metrics": {},
            "aggregate_metrics": agg,
            "cross_sections": cs_results,
            "default_cs_index": best_idx,
            "best_index": best_idx,
            "worst_index": worst_idx,
            "profile": default_cs.get("profile"),
            "image_name": image_name,
        }
        self.results_panel.set_results(results)
        return image

