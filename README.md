# TopoVein — IEEE IAMPro 2026

Topology-based biometric authentication using finger vein graph analysis
and persistent homology.

**Team:** Central University of Karnataka  
**Program:** IEEE CS Bangalore Chapter Internship 2026

## Pipeline

NIR Image → Preprocessing → Skeletonization → Graph G(V,E) → Persistent Homology → Hausdorff Matching → Auth Decision

## Repository Structure

| Folder | Owner | Purpose |
|--------|-------|---------|
| src/preprocessing | Samiksha | CLAHE, binarization |
| src/skeletonization | Harsh | Zhang-Suen thinning |
| src/graph_builder | Harsh | Node detection, graph G(V,E) |
| src/tda | Tarunima | Persistent homology |
| src/matching | Tarunima | Hausdorff distance matching |
| notebooks/ | All | Exploration notebooks |
| Dataset/ | Local only | See Dataset/README.md |

## Setup

```bash
pip install -r requirements.txt
```

## Dataset
FV-USM (123 subjects, 2 sessions, 4 fingers, 6 images each)
Not included — see Dataset/README.md

## Branch Structure
- main — stable, protected, merged only via PR
- samiksha — preprocessing module
- harsh — skeletonization and graph extraction
- tarunima — TDA and matching
- rishu — hardware integration

## Tech Stack
Python · OpenCV · scikit-image · NetworkX · NumPy · Matplotlib · giotto-tda · SciPy
