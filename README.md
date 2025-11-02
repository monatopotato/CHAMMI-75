# CHAMMI-75: pre-training multi-channel models with heterogeneous microscopy images

Vidit Agrawal<sup>1, 2</sup>, John Peters<sup>1, 2</sup>, Tyler Thompson<sup>1, 2</sup>, Mohammed Sanian<sup>3</sup>, Chau Pham<sup>4</sup>, Nikita Moshkov<sup>5</sup>, Arshad Kazi<sup>1, 2</sup>, Aditya Pillai<sup>1, 2</sup>, Jack Freeman<sup>1</sup>, Byunguk Kang<sup>6, 7</sup>, Samouil L. Farhi<sup>7</sup>, Ernest Fraenkel<sup>6</sup>, Ron Stewart<sup>1</sup>, Lassi Paavolainen<sup>3</sup>, Bryan Plummer<sup>4</sup>, Juan Caicedo<sup>1, 2</sup>

<sup>1</sup>Morgridge Institute for Research 
<sup>2</sup>University of Wisconsin-Madison  
<sup>3</sup>Institute for Molecular Medicine Finland (FIMM), University of Helsinki  
<sup>4</sup>Boston University  
<sup>5</sup>Institute of Computational Biology, Helmholtz Munich  
<sup>6</sup>Department of Biological Engineering, Massachusetts Institute of Technology  
<sup>7</sup>Spatial Technology Platform, Broad Institute of Harvard and MIT  

## Preprint Out!

Here is link for the preprint: [Link]

## Accessing the dataset

Please go to AWS and download the dataset from an S3 bucket:

Details, and steps

## Metadata Schema

The metadata comes in six major groups: **experiment**, **biology**, **imaging**, **microscopy**, **geometry**, and **storage** information. Each record in the metadata file points to a single channel file. The metadata is designed to facilitate grouping of channel files according to the categories described before. For each category, we have several metadata columns described below. 

> **Note:** If the information for an image is missing or not known, the corresponding value will be labeled with the string `"unknown"`. We try not to leave NaN or empty strings in the metadata file. If you see something, say something.

---

### Experiment

| Field | Description |
|-------|-------------|
| `experiment.study` | Identifier of the study. |
| `experiment.plate` | Plate where the image was acquired. If images come from another format (not plate based), this identifier can indicate a major group of experimental arrangements in the study. |
| `experiment.well` | Well position within the plate. The format of letter and number is preferred, but this is flexible. |
| `experiment.reagent` | Identifier or name of the treatment or reagent used to treat the cells. In many cases, this is a gene name, a compound name, or a protein name, while in other cases it may reflect other experimental intervention (e.g., temperature). |
| `experiment.control` | Whether the image comes from a control well or not, and what type of control they may be, for example, positive or negative control. If not a control, use the string `"no"`. |

### Biology

| Field | Description |
|-------|-------------|
| `biology.organism` | Name of the organism where the cells come from. For example, humans, mice, plants, etc. |
| `biology.cell_line` | Name of the cell line. Many cell lines have well known names (such as HeLa), other cell lines are from primary patients and have anonymized codes, and others from genetically modified organisms. |
| `biology.cell_type` | The functional type of cell, regardless of the cell line. Examples include neurons, red blood cells, cancer cells, pancreatic cells, etc. |

### Imaging

| Field | Description |
|-------|-------------|
| `imaging.multi_channel_id` | This is the field that ties together multiple channels. It is a consecutive number from the original database concatenated with the study number. A unique multi_channel_id connects the channels of an image. |
| `imaging.panel` | Names and dyes of the channels used to create the image. This gives context for where the observed channel file comes from. Example: `"DNA, protein, cytoplasm"`. |
| `imaging.channel` | Numeric value of the channel according to the panel. This value is one-based. |
| `imaging.channel_type` | Biological compartment of the cell that is visible in the channel. This is a list of standardized values that include: nucleus, cell body, bright-field, etc. |

### Microscopy

| Field | Description |
|-------|-------------|
| `microscopy.type` | Name of the type of microscopy used for acquisition of the channel file. Examples include: fluorescence, bright-field, confocal, cryoEM, etc. |
| `microscopy.magnification` | Numeric value of the magnification used to acquire the image. |
| `microscopy.fov` | Field of view, well site, or microscope position in the well when the channel was captured. |

### Geometry

| Field | Description |
|-------|-------------|
| `geometry.width` | Channel width in pixels. |
| `geometry.height` | Channel height in pixels. |
| `geometry.depth` | Total number of z-planes this channel belongs to, if the study is a 3D imaging assay. |
| `geometry.channels` | Total number of sibling channels in the same image. |
| `geometry.z_slice` | Number of the z-plane for this channel. It is a numerical value. |
| `geometry.timepoint` | Number of the frame in the timelapse sequence, if applicable. |

### Storage

| Field | Description |
|-------|-------------|
| `storage.filename` | Name of the zip file containing this channel. |

## SSL pre-training Comamnds

### Commands to run DINOv1

```bash
 python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch vit_small --data_path /scr/data/75ds_train/CHAMMI-75_train.zip --output_dir /scr/vidit/Models/test_3 --lr 0.00005 --batch_size_per_gpu 224 --guided_crops_path /scr/data/75ds_large_segmentations/CHAMMI-75_guidance.zip --multiscale True --dataset_size large --guided_cropping True
```

### Commands to run MAE

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --data_path /scr/data/CHAMMIv2s_train.zip --output_dir /scr/vidit/Models/MAE_75ds_baseline --batch_size 1024
```
