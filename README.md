# CHAMMI-75: pre-training multi-channel models with heterogeneous microscopy images

Vidit Agrawal<sup>1, 2</sup>, John Peters<sup>1, 2</sup>, Tyler Thompson<sup>1, 2</sup>, Mohammed Sanian<sup>3,4</sup>, Chau Pham<sup>5</sup>, Nikita Moshkov<sup>6</sup>, Arshad Kazi<sup>1, 2</sup>, Aditya Pillai<sup>1, 2</sup>, Jack Freeman<sup>1</sup>, Byunguk Kang<sup>7, 8</sup>, Samouil L. Farhi<sup>8</sup>, Ernest Fraenkel<sup>7</sup>, Ron Stewart<sup>1</sup>, Lassi Paavolainen<sup>3,4</sup>, Bryan Plummer<sup>5</sup>, Juan Caicedo<sup>1, 2</sup>

<sup>1</sup>Morgridge Institute for Research  
<sup>2</sup>University of Wisconsin-Madison  
<sup>3</sup>Institute for Molecular Medicine Finland (FIMM)  
<sup>4</sup>University of Helsinki  
<sup>5</sup>Boston University  
<sup>6</sup>Institute of Computational Biology, Helmholtz Munich  
<sup>7</sup>Massachusetts Institute of Technology  
<sup>8</sup>Broad Institute of MIT and Harvard

Official Github repository of CHAMMI-75: first of its kind 2.8 million multi-channel image dataset of microscopy imaging pooled from 75 different sources. The aim is to accelerate investigation of generalizable channel-agnostic foundation models in the field of microscopy.

## How to Cite

The work has been published at the International Conference on Learning Representations (ICLR) 2026: [Link](https://openreview.net/forum?id=SLjqdj3LPk)


```
@inproceedings{
agrawal2026chammi,
title={{CHAMMI}-75: pre-training multi-channel models with heterogeneous microscopy images},
author={Vidit Agrawal and John Peters and Tyler N. Thompson and Mohammad Vali Sanian and Chau Pham and Nikita Moshkov and Arshad Kazi and Aditya Pillai and Jack Freeman and Byunguk Kang and Samouil L. Farhi and Ernest Fraenkel and Ron M. Stewart and Lassi Paavolainen and Bryan A. Plummer and Juan C. Caicedo},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=SLjqdj3LPk}
}
```

## Accessing the dataset

Please go to AWS and download the dataset from an S3 bucket: https://registry.opendata.aws/chammi/

Command that will download the entire CHAMMI-75 project using aws cli (No AWS account required)

```bash
aws s3 ls --no-sign-request s3://chammi-data/
```

For more details and steps to download specific parts, go to [AWS-Download Instructions](./aws-tutorials)

We thank the AWS Open Data Sponsorship for hosting out dataset

## Running Benchmarks

Please see our [Benchmarks folders](./benchmarks) use our benchmarks!

