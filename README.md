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

Please go to AWS and download the dataset from an S3 bucket: https://registry.opendata.aws/chammi/

Command that will download the entire CHAMMI-75 project using aws cli (No AWS account required)

```bash
aws s3 ls --no-sign-request s3://chammi-data/
```

For more details and steps to download specific parts, go to [AWS-Download Instructions](./aws-tutorials)

We thank the AWS Open Data Sponsorship for hosting out dataset

## Running Benchmarks

Please see our [Benchmarks folders](./benchmarks) use our benchmarks!

