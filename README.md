# CHAMMI-75: pre-training multi-channel models with heterogeneous microscopy images

Vidit Agrawal<sup>1, 2</sup>, John Peters<sup>1, 2</sup>, Tyler Thompson<sup>1, 2</sup>, Mohammed Sanian<sup>3</sup>, Chau Pham<sup>4</sup>, Nikita Moshkov<sup>5</sup>, Arshad Kazi<sup>1, 2</sup>, Aditya Pillai<sup>1, 2</sup>, Jack Freeman<sup>1</sup>, Byunguk Kang<sup>6, 7</sup>, Samouil L. Farhi<sup>7</sup>, Ernest Fraenkel<sup>6</sup>, Ron Stewart<sup>1</sup>, Lassi Paavolainen<sup>3</sup>, Bryan Plummer<sup>4</sup>, Juan Caicedo<sup>1, 2</sup>

<sup>1</sup>Morgridge Institute for Research 
<sup>2</sup>University of Wisconsin-Madison  
<sup>3</sup>Institute for Molecular Medicine Finland (FIMM), University of Helsinki  
<sup>4</sup>Boston University  
<sup>5</sup>Institute of Computational Biology, Helmholtz Munich  
<sup>6</sup>Department of Biological Engineering, Massachusetts Institute of Technology  
<sup>7</sup>Spatial Technology Platform, Broad Institute of Harvard and MIT  

Official Github repository of CHAMMI-75: first of its kind 2.8 million multi-channel image dataset of microscopy imaging pooled from 75 different sources. The aim is to accelerate investigation of generalizable channel-agnostic foundation models in the field of microscopy.

## Preprint Out Soon!

Here is link for the preprint: [Link]

## Accessing the dataset

Please go to AWS and download the dataset from an S3 bucket:

Details, and steps

## Running Benchmarks

Please see our tutorials in aws-tutorials folder to download and use our benchmarks present in the benchmarks folder!

## SSL pre-training Comamnds

### Commands to run DINOv1

```bash
 python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch vit_small --data_path /scr/data/75ds_train/CHAMMI-75_train.zip --output_dir /scr/vidit/Models/test_3 --lr 0.00005 --batch_size_per_gpu 224 --guided_crops_path /scr/data/75ds_large_segmentations/CHAMMI-75_guidance.zip --multiscale True --dataset_size large --guided_cropping True
```

### Commands to run MAE

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --data_path /scr/data/CHAMMIv2s_train.zip --output_dir /scr/vidit/Models/MAE_75ds_baseline --batch_size 1024
```

