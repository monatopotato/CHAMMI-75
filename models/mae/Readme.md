## SSL Pre-training Commands


### Commands to run MAE

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --data_path /scr/data/CHAMMIv2s_train.zip --output_dir /scr/vidit/Models/MAE_75ds_baseline --batch_size 1024
```
