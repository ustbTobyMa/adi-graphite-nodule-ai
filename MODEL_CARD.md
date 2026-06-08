# Model Card: Graphite U-Net

## Model

- Name: `graphite_unet_seed7`
- Architecture: compact 2D U-Net
- Inputs: grayscale SEM image
- Outputs: graphite nodule probability map
- Released formats: PyTorch checkpoint (`.pt`) and ONNX model (`.onnx`)

## Intended Use

- Segment graphite nodules in ductile cast iron SEM images
- Extract nodule count density, equivalent diameter, circularity, and area fraction
- Support reproducible reviewer inspection through overlays and downloadable metrics

## Public Example Policy

Only one SEM example and its reference overlay are included in this public repository. The broader manuscript image set is not released here to avoid unnecessary image-asset disclosure.

## Training Data Summary

The model was trained on curated low-magnification SEM fields from the manuscript study. High-magnification, close-up, edge/defect, and qualitative fields were excluded from quantitative field-level training/statistics.

The training masks were curated pseudo masks, so the held-out Dice values should be interpreted as agreement with curated pseudo-labels rather than a manual expert-label benchmark.

## Quality Guidance

Users should inspect overlays for each new dataset. If the model segments scratches, pores, matrix contrast, edge artifacts, or scale bars as nodules, adjust the threshold or object-size filters and report those settings.

## Out-of-Scope Use

- Unchecked use on unrelated materials systems
- High-magnification close-ups treated as field-level count statistics
- Defect, edge, fracture, or scale-bar regions used as representative fields
- Fully automated reporting without visual overlay inspection
