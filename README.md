# Graphite Nodule Analyzer

AI-assisted segmentation and quantitative analysis of graphite nodules in ductile iron microscopy images.

This repository provides a trained CNN-based computer-vision workflow, a browser app, and a Python batch-analysis script. The browser app is designed for quick single-image analysis, while the Python workflow is intended for reproducible batch processing.

## Use In Browser

Open the GitHub Pages app:

https://ustbTobyMa.github.io/adi-graphite-nodule-ai/

The browser app runs locally with ONNX Runtime Web. Images uploaded to the browser are processed on the user's computer and are not sent to a remote server. A sample image is loaded automatically, so users can click **Run analysis** immediately or upload their own image.

Users can:

- upload a microscopy image
- use common browser inputs including PNG, JPG/JPEG, BMP, WebP, TIF, and TIFF
- enter the microscopy pixel size in μm per source pixel, or calibrate it by selecting the two ends of a scale bar in the image
- run the released CNN-based computer-vision workflow in the browser
- inspect the segmentation overlay and synchronized equivalent-diameter and circularity histograms
- download image-level and object-level CSV outputs

The main browser outputs are nodule count, count density in mm⁻², mean equivalent diameter in μm, and mean circularity. Circularity is calculated as `4πA/P²` and should not be interpreted as formal ASTM A247 nodularity. Multi-page TIFF files are loaded from the first page.

## Scientific Validation

The released segmentation model was checked against curated validation masks and a public reference workflow example.

| Metric | Value | Scope |
| --- | ---: | --- |
| Dice coefficient | 0.963 | Held-out curated reference masks |
| IoU | 0.929 | Held-out curated reference masks |
| Precision | 0.933 | Held-out curated reference masks |
| Recall | 0.995 | Held-out curated reference masks |
| Nodule-count MAE | 2.33 nodules | Two-seed repeatability over low-magnification fields |
| Count-density MAE | 2.19 mm⁻² | Two-seed repeatability over low-magnification fields |
| Area-fraction difference | 0.20 percentage points | Public reference-workflow check |

### Public Example: Reference Workflow vs AI Workflow

The public example was compared with a traditional ImageJ-style reference workflow: inverted-grayscale Otsu thresholding, watershed splitting, the same 1.16279 μm/pixel scale, and the same 25-20000 px object-size filters.

| Metric | Reference workflow | AI workflow | Relative difference |
| --- | ---: | ---: | ---: |
| Nodule count | 180 | 178 | 1.1% |
| Count density | 169.28 mm⁻² | 167.40 mm⁻² | 1.1% |
| Mean equivalent diameter | 25.81 μm | 25.75 μm | 0.2% |
| Mean circularity | 0.791 | 0.851 | 7.5% |

Relative difference is calculated against the reference-workflow result. This is a workflow-level validation check using a public example and is not an independent multi-expert ASTM A247 benchmark. The slightly larger difference in mean circularity mainly reflects boundary-smoothing and watershed-splitting differences between threshold-based segmentation and probability-map-based CNN segmentation; count density and equivalent diameter remain highly consistent.

The reported segmentation metrics quantify agreement with curated validation masks rather than a fully independent expert-labeled benchmark.

## Batch Analysis With Python

For multiple images, reproducible outputs, masks, overlays, JSON summaries, and object-level CSV files, use the Python script:

```bash
git clone https://github.com/ustbTobyMa/adi-graphite-nodule-ai.git
cd adi-graphite-nodule-ai

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/graphite_nodule_analyzer.py \
  --image-dir /path/to/microscopy_images \
  --checkpoint models/graphite_unet_seed7.pt \
  --output-dir /path/to/output \
  --pixel-size-um 1.16279 \
  --threshold 0.50 \
  --min-area-px 25 \
  --max-area-px 20000
```

`--pixel-size-um` must be measured from the user's own microscopy scale bar. For example, if a 100 μm scale bar spans 86 pixels, use `100/86 = 1.1627906976744187`. The example values above reproduce the public example configuration.

## Outputs

The Python output directory contains:

- `summary_report.json`
- `image_summary.csv`
- `component_measurements.csv`
- `masks/`
- `probability/`
- `overlays/`
- `overlay_contact_sheet.png`

The browser app provides CSV downloads for the current image.

## Released Files

```text
index.html                         Browser app
models/graphite_unet_seed7.onnx    Browser model
models/graphite_unet_seed7.pt      Python model
scripts/graphite_nodule_analyzer.py
examples/images/sample_ductile_iron_sem.png
examples/reference/sample_overlay.png
examples/reference/sample_mask.png
```

## Recommended Image Input

Use images similar to the training domain:

- polished ductile cast iron microscopy fields
- representative low-magnification fields for field-level statistics
- graphite nodules visible as dark features in a lighter matrix

Avoid high-magnification close-ups, edge regions, fracture/defect fields, scale bars inside the analysis area, and images with strong charging or preparation artifacts.

## Limitations

The released model was trained on curated ductile-iron microscopy images from the associated study. It should be visually checked on new microscopes, magnifications, etching conditions, alloy systems, and image acquisition settings. For formal reporting, inspect the exported overlays and report the pixel size, threshold, and object-size filters.

## Citation

If you use this workflow, please cite the associated manuscript and this repository.
