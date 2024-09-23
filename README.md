## Description

This repository is a fork of [original PURE](https://github.com/maxdreyer/PURE).

### Abstract 

XXXX

## Table of Contents

  - [Description](#description)
  - [Abstract](#abstract)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Config Files](#config-files)
  - [Preprocessing](#preprocessing)
  - [Render UMAP and Feature Visualizations for PURE](#render-umap-and-feature-visualizations-for-pure)
  - [Evaluate Interpretability of Disentangled Features](#evaluate-interpretability-of-disentangled-features)
  - [Evaluate Alignment to CLIP](#evaluate-alignment-to-clip)
  - [Citation](#citation)

## Installation

We use Python 3.8.10. To install the required packages, run:

```bash 
pip install -r requirements.txt
```

Secondly, we need to download the **ImageNet** dataset. To do so, visit
the [ImageNet website](https://image-net.org/download.php) and download the training and validation images.

### Config Files

Please adapt the config files to your setup (e.g., correct dataset paths).
To do so,
specify the config files in `configs/imagenet/*.yaml` (replace `*` with model name):

## Preprocessing

XXX

## Render UMAP and Feature Visualizations for PURE

XXX

## Evaluate Interpretability of Disentangled Features

XXX

## Evaluate Alignment to CLIP

XXX

## Citation

Cite the following work, if used in your research:

```bibtex
@article{dreyer2024pure,
  title={PURE: Turning Polysemantic Neurons Into Pure Features by Identifying Relevant Circuits},
  author={Dreyer, Maximilian and Purelku, Erblina and Vielhaben, Johanna and Samek, Wojciech and Lapuschkin, Sebastian},
  journal={arXiv preprint arXiv:2404.06453},
  year={2024}
}
```