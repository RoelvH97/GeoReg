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

Example registration optimization showing convergence for both biplanar views:

**Posteroanterior (PA) View**
![PA View](assets/optimization_progress_pa.gif)

**Lateral (L) View**
![L View](assets/optimization_progress_l.gif)

## Citation

If you find this work useful, please cite our submission:

```bibtex
@misc{vanherten2025georeg,
  title={Direct biplanar DSA-to-CTA registration with geodesic consistency for acute ischemic stroke},
  author={van Herten, Rudolf L. M. and Graf, Robert and Bitzer, Felix and Kirschke, Jan S. and Paetzold, Johannes C.},
  note={Submitted to Medical Imaging with Deep Learning (MIDL) 2026},
  year={2025}
}
```

## License

© 2025 CC-BY 4.0
