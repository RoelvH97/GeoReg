# GeoReg: Direct Biplanar DSA-to-CTA Registration

This repository contains the implementation for the MIDL 2026 submission:

**"Direct biplanar DSA-to-CTA registration with geodesic consistency for acute ischemic stroke"**
*Rudolf L. M. van Herten, Robert Graf, Felix Bitzer, Jan S. Kirschke, Johannes C. Paetzold*

## Overview

GeoReg provides a direct approach to registering intraoperative Digital Subtraction Angiography (DSA) with pre-procedural Computed Tomography Angiography (CTA) for acute ischemic stroke imaging, **without requiring vessel segmentation**.

### Key Features

- **Segmentation-free registration**: Uses maximum intensity projections (MAP) from DSA sequences instead of vessel segmentations
- **Biplanar optimization**: Jointly optimizes posteroanterior (PA) and lateral (L) views with geodesic consistency constraints
- **Differentiable rendering**: Leverages DiffDRR for efficient gradient-based pose estimation

### Method

The approach recovers a silhouette of the subtracted X-ray using temporal maximum intensity projections, enabling direct image-similarity-based registration between DSA and CTA. A soft geodesic consistency constraint maintains approximately orthogonal biplanar geometry while accommodating real-world scanner configurations.

## Optimization Progress

Example registration optimization showing convergence:

![Optimization Progress](outputs/2025-12-02/11-43-17/sub-stroke0001/pre/optimization_progress.mp4)

## Citation

```bibtex
@inproceedings{vanherten2026georeg,
  title={Direct biplanar DSA-to-CTA registration with geodesic consistency for acute ischemic stroke},
  author={van Herten, Rudolf L. M. and Graf, Robert and Bitzer, Felix and Kirschke, Jan S. and Paetzold, Johannes C.},
  booktitle={Medical Imaging with Deep Learning},
  year={2026}
}
```

## License

© 2026 CC-BY 4.0
