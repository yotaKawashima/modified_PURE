# find maximally activating inputs.
python3 -m experiments.preprocessing.crp_run --config_file "configs/imagenette2/resnet34_timm.yaml"

# 
for attribution_layer_name in 'block_0' 'block_1' 'block_2'; do
    python3 -m experiments.preprocessing.compute_latent_features_choose_attribution_layer --config_file "configs/imagenette2/resnet34_timm.yaml" --attribution_layer_name $attribution_layer_name
done

# 
python3 -m experiments.preprocessing.compute_embeddings --config_file "configs/imagenette2/resnet34_timm.yaml" 