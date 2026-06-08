# Graphite Nodule AI

AI-assisted graphite nodule segmentation and statistics for ductile cast iron SEM images.

This repository provides a trained U-Net model, a browser demo, and a Python batch-analysis script. One public SEM example and reference overlay are provided for testing the workflow.

## Use In Browser

Open the GitHub Pages app:

https://ustbTobyMa.github.io/adi-graphite-nodule-ai/

The browser app runs locally in the user's browser with ONNX Runtime Web. Uploaded SEM images are not sent to a server. Users can:

- upload a SEM image
- use common browser inputs including PNG, JPG/JPEG, BMP, WebP, TIF, and TIFF
- enter the SEM pixel size in μm per source pixel, or calibrate it by selecting the two ends of a scale bar in the image
- run the released U-Net model in the browser
- inspect the overlay and synchronized equivalent-diameter and spheroidicity histograms
- download image-level and object-level CSV outputs

For manuscript-level statistics, we used the Python batch workflow with fixed pixel size, probability threshold, watershed splitting, and object-size filters. The browser tool is intended for reviewer inspection and single-image verification. Multi-page TIFF files are loaded from the first page.

## Reviewer Validation Summary

### Public Example: AI vs ImageJ-Style Workflow

The public example was compared with a traditional ImageJ-style workflow: inverted-grayscale Otsu thresholding, watershed splitting, the same 1.16279 μm/pixel scale, and the same 25-20000 px object-size filters.

| Metric | ImageJ-style workflow | Python U-Net batch | Relative error |
| --- | ---: | ---: | ---: |
| Nodule count | 180 | 178 | 1.1% |
| Count density | 169.28 mm⁻² | 167.40 mm⁻² | 1.1% |
| Mean diameter | 25.81 μm | 25.76 μm | 0.2% |
| Mean spheroidicity | 0.791 | 0.851 | 7.6% |

Relative error is calculated against the ImageJ-style result. This is a public single-image sanity check rather than a fully independent expert-label benchmark. The U-Net values in this table are from the Python batch workflow used for formal statistics.

### Model and Workflow Metrics

| Metric | Value | Scope |
| --- | ---: | --- |
| Dice coefficient | 0.963 | Held-out curated reference masks |
| IoU | 0.929 | Held-out curated reference masks |
| Precision / recall | 0.933 / 0.995 | Held-out curated reference masks |
| Nodule-count MAE | 2.33 nodules | Two-seed repeatability over low-magnification fields |
| Count-density MAE | 2.19 mm⁻² | Two-seed repeatability over low-magnification fields |
| Area-fraction difference | 0.20 percentage points | Public ImageJ-style check |

The reported validation metrics quantify agreement with curated reference masks rather than a fully independent expert-labeled benchmark.

## Batch Analysis With Python

For formal or batch analysis, use the Python script:

```bash
git clone https://github.com/ustbTobyMa/adi-graphite-nodule-ai.git
cd adi-graphite-nodule-ai

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/graphite_nodule_analyzer.py \
  --image-dir /path/to/sem_images \
  --checkpoint models/graphite_unet_seed7.pt \
  --output-dir /path/to/output \
  --pixel-size-um YOUR_MICROMETRES_PER_PIXEL
```

`--pixel-size-um` must be measured from the user's own SEM scale bar. For example, if a 100 μm scale bar spans 86 pixels, use `100/86 = 1.1627906976744187`.

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
```

## Recommended Image Input

Use images similar to the training domain:

- polished ductile cast iron SEM fields
- representative low-magnification fields for field-level statistics
- graphite nodules visible as dark features in a lighter matrix

Avoid high-magnification close-ups, edge regions, fracture/defect fields, scale bars inside the analysis area, and images with strong charging or preparation artifacts.

## Limitations

The released model was trained on curated SEM images from the manuscript study. It should be visually checked on new microscopes, magnifications, etching conditions, alloy systems, and image acquisition settings. For formal reporting, inspect the exported overlays and report the pixel size, threshold, and object-size filters.

## Citation

If you use this workflow, please cite the associated manuscript and this repository.
