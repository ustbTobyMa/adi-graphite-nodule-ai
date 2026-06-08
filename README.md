# Graphite Nodule AI

AI-assisted graphite nodule segmentation and statistics for ductile cast iron SEM images.

This repository provides a trained U-Net model, a browser demo, and a Python batch-analysis script. The public example set is intentionally minimal: one SEM image and one reference overlay are included so users can test the workflow without exposing the full manuscript image assets.

## Use In Browser

Open the GitHub Pages app:

https://ustbTobyMa.github.io/adi-graphite-nodule-ai/

The browser app runs locally in the user's browser with ONNX Runtime Web. Uploaded SEM images are not sent to a server. Users can:

- upload a SEM image
- enter the SEM pixel size in micrometres per pixel
- run the released U-Net model in the browser
- inspect the overlay
- download image-level and object-level CSV outputs

The browser app is intended for quick inspection and lightweight use.

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

`--pixel-size-um` must be measured from the user's own SEM scale bar. For example, if a 100 um scale bar spans 86 pixels, use `100/86 = 1.1627906976744187`.

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
