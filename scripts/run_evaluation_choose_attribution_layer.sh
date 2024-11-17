
experiment=experiments.disentangling.eval_CLIP_alignment_choose_attribution_layer

for n_clusters in {2..5}; do
    for attribution_layer_name in 'block_0' 'block_1' 'block_2'; do
        python3 -m $experiment --config_file "configs/imagenette2/resnet34_timm.yaml" --attribution_layer_name $attribution_layer_name --num_clusters $n_clusters
    done
done
