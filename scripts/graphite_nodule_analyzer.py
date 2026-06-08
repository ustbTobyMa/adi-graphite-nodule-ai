#!/usr/bin/env python3
"""Analyze spheroidal graphite nodules in ductile cast iron SEM images.

The script loads the released U-Net checkpoint, segments graphite nodules, and
exports image-level and object-level morphology statistics. Users must provide
the physical pixel size from their own SEM scale bar.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
from PIL import Image
from skimage import measure, morphology, segmentation

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.down1 = DoubleConv(1, c)
        self.down2 = DoubleConv(c, c * 2)
        self.down3 = DoubleConv(c * 2, c * 4)
        self.down4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c * 8, c * 16)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.conv4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.conv3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.conv2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.conv1 = DoubleConv(c * 2, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        b = self.bottleneck(self.pool(d4))
        x = self.up4(b)
        x = self.conv4(torch.cat([x, d4], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, d3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, d2], dim=1))
        x = self.up1(x)
        x = self.conv1(torch.cat([x, d1], dim=1))
        return self.out(x)


def image_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_channels = int(checkpoint.get("config", {}).get("base_channels", 32))
    model = UNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_probability(model: UNet, gray: np.ndarray, device: torch.device) -> np.ndarray:
    image = gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image[None, None, :, :]).to(device)
    logits = model(tensor)
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


def postprocess_mask(probability: np.ndarray, threshold: float, min_object_px: int) -> np.ndarray:
    mask = probability > threshold
    mask = morphology.remove_small_objects(mask, max_size=min_object_px - 1)
    mask = morphology.remove_small_holes(mask, max_size=39)
    mask = morphology.closing(mask, morphology.disk(2))
    return mask.astype(bool)


def split_touching_nodules(mask: np.ndarray) -> np.ndarray:
    distance = ndi.distance_transform_edt(mask)
    local_max = morphology.local_maxima(distance)
    markers = measure.label(local_max)
    if markers.max() == 0:
        return measure.label(mask)
    return segmentation.watershed(-distance, markers, mask=mask)


def component_rows(labels: np.ndarray, pixel_size_um: float, min_object_px: int, max_object_px: int) -> list[dict]:
    rows: list[dict] = []
    h, w = labels.shape
    for region in measure.regionprops(labels):
        minr, minc, maxr, maxc = region.bbox
        area = float(region.area)
        if area < min_object_px or area > max_object_px:
            continue
        touches_border = minr <= 1 or minc <= 1 or maxr >= h - 1 or maxc >= w - 1
        if touches_border:
            continue
        perimeter = max(float(region.perimeter), 1e-9)
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if float(region.solidity) < 0.45:
            continue
        rows.append(
            {
                "label": int(region.label),
                "area_px": area,
                "area_um2": area * pixel_size_um**2,
                "perimeter_px": perimeter,
                "equivalent_diameter_px": float(region.equivalent_diameter_area),
                "equivalent_diameter_um": float(region.equivalent_diameter_area) * pixel_size_um,
                "major_axis_length_um": float(region.axis_major_length) * pixel_size_um,
                "minor_axis_length_um": float(region.axis_minor_length) * pixel_size_um,
                "eccentricity": float(region.eccentricity),
                "solidity": float(region.solidity),
                "circularity": circularity,
                "centroid_y": float(region.centroid[0]),
                "centroid_x": float(region.centroid[1]),
                "bbox_min_row": int(minr),
                "bbox_min_col": int(minc),
                "bbox_max_row": int(maxr),
                "bbox_max_col": int(maxc),
            }
        )
    return rows


def mask_from_rows(labels: np.ndarray, rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.zeros_like(labels, dtype=bool)
    keep = np.array([row["label"] for row in rows], dtype=labels.dtype)
    return np.isin(labels, keep)


def summarize_image(name: str, gray: np.ndarray, mask: np.ndarray, rows: list[dict], pixel_size_um: float) -> dict:
    field_area_mm2 = gray.shape[0] * gray.shape[1] * pixel_size_um**2 / 1_000_000.0
    diameters = np.array([row["equivalent_diameter_um"] for row in rows], dtype=float)
    circularities = np.array([row["circularity"] for row in rows], dtype=float)
    count = len(rows)
    return {
        "image": name,
        "height_px": gray.shape[0],
        "width_px": gray.shape[1],
        "pixel_size_um": pixel_size_um,
        "field_area_mm2": field_area_mm2,
        "nodule_count": count,
        "nodule_count_per_mm2": count / field_area_mm2 if field_area_mm2 else np.nan,
        "graphite_area_fraction": float(mask.sum() / mask.size),
        "mean_equivalent_diameter_um": float(np.mean(diameters)) if count else np.nan,
        "sd_equivalent_diameter_um": float(np.std(diameters, ddof=1)) if count > 1 else np.nan,
        "mean_circularity": float(np.mean(circularities)) if count else np.nan,
        "sd_circularity": float(np.std(circularities, ddof=1)) if count > 1 else np.nan,
        "nodularity_circularity_ge_0_6_pct": float(np.mean(circularities >= 0.6) * 100.0) if count else np.nan,
        "nodularity_circularity_ge_0_7_pct": float(np.mean(circularities >= 0.7) * 100.0) if count else np.nan,
    }


def draw_overlay(gray: np.ndarray, mask: np.ndarray, rows: list[dict], output_path: Path) -> None:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    overlay = rgb.copy()
    overlay[mask, 0] = 1.0
    overlay[mask, 1] *= 0.25
    overlay[mask, 2] *= 0.25
    blended = 0.65 * rgb + 0.35 * overlay

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=180)
    ax.imshow(blended, cmap="gray")
    ax.contour(mask, colors=[(0.0, 0.9, 1.0)], linewidths=0.35)
    for row in rows:
        ax.plot(row["centroid_x"], row["centroid_y"], ".", color="yellow", markersize=1.1)
    ax.set_axis_off()
    ax.set_title(f"{output_path.stem}: {len(rows)} nodules", fontsize=8)
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(overlay_dir: Path, output_path: Path, max_images: int = 16) -> None:
    paths = sorted(overlay_dir.glob("*_overlay.png"))[:max_images]
    if not paths:
        return
    cols = 4
    rows = math.ceil(len(paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 1.9), dpi=180)
    axes_arr = np.array(axes).reshape(-1)
    for ax, path in zip(axes_arr, paths):
        ax.imshow(np.array(Image.open(path)))
        ax.set_title(path.name.replace("_overlay.png", ""), fontsize=6)
        ax.set_axis_off()
    for ax in axes_arr[len(paths) :]:
        ax.set_axis_off()
    fig.tight_layout(pad=0.35)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-assisted graphite nodule statistics for ductile cast iron SEM images.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing SEM images.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Released U-Net checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for masks, overlays, and CSV outputs.")
    parser.add_argument("--pixel-size-um", type=float, required=True, help="SEM pixel size in micrometres per pixel.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for graphite mask.")
    parser.add_argument("--min-object-px", type=int, default=25, help="Minimum nodule area in pixels.")
    parser.add_argument("--max-object-px", type=int, default=20000, help="Maximum nodule area in pixels.")
    args = parser.parse_args()

    if args.pixel_size_um <= 0:
        raise ValueError("--pixel-size-um must be positive. Measure it from your SEM scale bar.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["masks", "overlays", "probability"]:
        (args.output_dir / subdir).mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    summaries: list[dict] = []
    objects: list[dict] = []

    for path in image_paths(args.image_dir):
        gray = np.array(Image.open(path).convert("L"))
        probability = predict_probability(model, gray, device)
        raw_mask = postprocess_mask(probability, args.threshold, args.min_object_px)
        labels = split_touching_nodules(raw_mask)
        rows = component_rows(labels, args.pixel_size_um, args.min_object_px, args.max_object_px)
        mask = mask_from_rows(labels, rows)
        for row in rows:
            row["image"] = path.name
        summary = summarize_image(path.name, gray, mask, rows, args.pixel_size_um)
        summaries.append(summary)
        objects.extend(rows)

        Image.fromarray((probability * 255).astype(np.uint8)).save(args.output_dir / "probability" / f"{path.stem}_probability.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(args.output_dir / "masks" / f"{path.stem}_mask.png")
        draw_overlay(gray, mask, rows, args.output_dir / "overlays" / f"{path.stem}_overlay.png")

    write_csv(args.output_dir / "image_summary.csv", summaries)
    write_csv(args.output_dir / "component_measurements.csv", objects)
    make_contact_sheet(args.output_dir / "overlays", args.output_dir / "overlay_contact_sheet.png")

    count_density = np.array([row["nodule_count_per_mm2"] for row in summaries], dtype=float)
    diameter = np.array([row["mean_equivalent_diameter_um"] for row in summaries], dtype=float)
    circularity = np.array([row["mean_circularity"] for row in summaries], dtype=float)
    report = {
        "processed_images": len(summaries),
        "pixel_size_um": args.pixel_size_um,
        "threshold": args.threshold,
        "nodule_count_density_mean_per_mm2": float(np.nanmean(count_density)),
        "nodule_count_density_sd_per_mm2": float(np.nanstd(count_density, ddof=1)) if len(count_density) > 1 else 0.0,
        "equivalent_diameter_mean_um": float(np.nanmean(diameter)),
        "equivalent_diameter_sd_um": float(np.nanstd(diameter, ddof=1)) if len(diameter) > 1 else 0.0,
        "circularity_mean": float(np.nanmean(circularity)),
        "circularity_sd": float(np.nanstd(circularity, ddof=1)) if len(circularity) > 1 else 0.0,
    }
    with (args.output_dir / "summary_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
