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

## Reviewer-Facing Validation

### Public Example: Traditional Workflow Comparison

The public SEM example was checked against a traditional ImageJ-style workflow: inverted-grayscale Otsu thresholding, watershed splitting, the same 1.16279 μm/pixel scale, and the same 25-20000 px object-size filters.

| Metric | ImageJ-style workflow | Python U-Net batch | Relative error |
| --- | ---: | ---: | ---: |
| Nodule count | 180 | 178 | 1.1% |
| Count density | 169.28 mm⁻² | 167.40 mm⁻² | 1.1% |
| Mean diameter | 25.81 μm | 25.76 μm | 0.2% |
| Mean spheroidicity | 0.791 | 0.851 | 7.6% |

Relative error is calculated against the ImageJ-style result. The U-Net values are from the Python batch workflow used for formal statistics. This single-image check is intended to make the released workflow auditable, not to replace a fully independent expert-label benchmark.

### Performance Metrics

| Metric | Value | Scope |
| --- | ---: | --- |
| Dice coefficient | 0.963 | Held-out curated reference masks |
| IoU | 0.929 | Held-out curated reference masks |
| Precision / recall | 0.933 / 0.995 | Held-out curated reference masks |
| Nodule-count MAE | 2.33 nodules | Two-seed repeatability over low-magnification fields |
| Count-density MAE | 2.19 mm⁻² | Two-seed repeatability over low-magnification fields |
| Equivalent-diameter MAE | 0.25 μm | Two-seed repeatability over low-magnification fields |
| Area-fraction difference | 0.20 percentage points | Public ImageJ-style check |

The reported validation metrics quantify agreement with curated reference masks rather than a fully independent expert-labeled benchmark.

## Quality Guidance

Users should inspect overlays for each new dataset. If the model segments scratches, pores, matrix contrast, edge artifacts, or scale bars as nodules, adjust the threshold or object-size filters and report those settings.

## Out-of-Scope Use

- Unchecked use on unrelated materials systems
- High-magnification close-ups treated as field-level count statistics
- Defect, edge, fracture, or scale-bar regions used as representative fields
- Fully automated reporting without visual overlay inspection
