# HPA Sub-Cell Evaluation

## For running hpa features extraction

```
accelerate launch --multi_gpu --num_processes=8 accelerate_hpa_features.py
```

## For running protein localization evaluations in mini-hpa

```
python train_classification.py -f {saving_metrics_locations} -cc locations -uc all_unique_cats

python train_classification.py -f {saving_metrics_locations} -cc locations -uc challenge_cats
```

Note: Saving_metrics_locations needs to be the same folder where the features were extracted out!
