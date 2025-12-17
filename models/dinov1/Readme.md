## SSL pre-training Commands



### Commands to run DINOv1
```bash
 python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch vit_small --data_path /scr/data/75ds_train/CHAMMI-75_train.zip --output_dir /scr/vidit/Models/test_3 --lr 0.00005 --batch_size_per_gpu 224 --guided_crops_path /scr/data/75ds_large_segmentations/CHAMMI-75_guidance.zip --multiscale True --dataset_size large --guided_cropping True
```
