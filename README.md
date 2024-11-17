## Description

This repository is a fork of the original [PURE](https://github.com/maxdreyer/PURE). Unlike the original version, you can select attribution layers by using codes in this fork.

[Report](https://github.com/yotaKawashima/modified_PURE/blob/main/Kawashima_LabRotation_report.pdf).


## Installation

You need Python 3.8.10. To install the required packages, run:

```bash 
pip install -r requirements.txt
```

You also need the **ImageNet** or **Imagenette** dataset. Download dataset from the following websites: 
- ImageNet website: [link](https://image-net.org/download.php)
- Imagenette website: [link](https://github.com/fastai/imagenette)

### Config Files

Before running the code, update the configuration files in the `configs` directory to match your setup (e.g., correct dataset paths).

For example, if you want to use the Imagenette 2 dataset for ResNet34, update `"configs/imagenette2/resnet34_timm.yaml"`.

## Preprocessing

The following command collects maximally activating images, disentangles neurons, and computes embeddings in different spaces (e.g., CLIP):

```bash 
bash scripts/run_preprocessing_choose_attribution_layer.sh
```

In the shell script, you can select the layer where you want to disentangle neurons (target layer) and the layer where you compute relevance scores for the disentanglement (attribution layer). Specifically, you can pass two optional arguments to `compute_latent_features_choose_attribution_layer`: `layer_name` for specifying the target layer and `attribution_layer_name` for the attribution layer.

For example, if you want to disentangle neurons in `block_3` using relevance scores in `block_1`, run:

```bash 
python3 -m experiments.preprocessing.compute_latent_features_choose_attribution_layer --config_file "configs/imagenette2/resnet34_timm.yaml" --layer_name 'block_3' --attribution_layer_name 'block_1'
```

## Render UMAP and Feature Visualizations for PURE 

You can qualitatively evaluate disentanglement by visualising maximally activating images in relevance score space. Better disentanglement should result in better image clusters in the relevance score space. The following command plots images in different spaces:

```bash
python -m experiments.plotting.plot_neurons_diff_layers --config_file "config_file" --neurons $neurons --layer_name $target_layer
```

For example, if you want to disentangle neuron 4, 327, 101, 94, and 100 in block_3, 
```bash
python -m experiments.plotting.plot_neurons_diff_layers --config_file "configs/imagenette2/resnet34_timm.yaml" --neurons "4,327,101,94,100" --layer_name "block_3"
```

## Evaluate Alignment to CLIP

You can quantitativelly evaluate disentanglement by checking correlation between embeddings in CLIP space and in relevance score space. You can do this analysis in `experiments/disentangling/distance_correlation_choose_attribution_layer.ipynb`.

## Evaluate Interpretability of Disentangled Features (optional and under construction)

Quantitative evaluation can also be done by checking distances in each embedding space. The following command computes distances:

```bash
bash scripts/run_evaluation_choose_attribution_layer.sh
```

Note that you need to specify the attribution layer in the shell script. After running the script, you can visualize the result in `experiments/disentangling/distance_correlation_choose_attribution_layer.ipynb`.


