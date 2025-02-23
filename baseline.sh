for dataset in adni_AAL116  # abide_full_AAL116
do
for wd in 5e-3
do
  for l in 4
  do
    for lr in 5e-5
  do
        model="configs/"${dataset}"/TUs_graph_classification_BrainPromptG_"${dataset}"_100k.json"
        echo ${model}
        python3 llm-main.py --gpu_id 0 --node_feat_transform pearson --max_time 60 --config $model --dropout 0.5 --L $l --edge_ratio 0.2 --lambda1 1.0 --init_lr $lr --min_lr 1e-6 --weight_decay $wd
  done
done
done
done
