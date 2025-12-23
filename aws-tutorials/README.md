# Downloading the dataset from AWS S3 bucket


The entire dataset with the test sets can be downloaded using the following command. Please ensure you have the AWS CLI installed. Also, please make sure you have around 5TB of free space on your local machine before running the command below.

## Downloading the entire package
```bash
aws s3 sync --no-sign-request s3://chammi-data/ ./local-directory/
```

If you want to download the 6 benchmarks only (test sets). You can use the following commands:

## Downloading the CHAMMI-75 test set
```bash
aws s3 sync --no-sign-request s3://chammi-data/CHAMMI-75/CHAMMI-75_test/ ./local-directory/
```

## Downloading the entire 2.8 million images CHAMMI-75 training set
If you want to download only the CHAMMI-75 training set. You can use the following command:

```bash
aws s3 sync --no-sign-request s3://chammi-data/CHAMMI-75/CHAMMI-75_train/ ./local-directory/
aws s3 sync --no-sign-request s3://chammi-data/CHAMMI-75/CHAMMI-75_train_metadata.csv ./local-directory/
aws s3 sync --no-sign-request s3://chammi-data/CHAMMI-75/CHAMMI-75_guidance/ ./local-directory/
```


## Downloading the CHAMMI-75 small training set
If you want to download the CHAMMI-75 small training set. You can use the python script `download_chammi_small.py` provided in the repository:

```bash
python download_chammi_small.py --download-folder /local-directory/ --workers 16 --guidance
```

Flags for the script:
- `--download-folder`: Path to the folder where you want to download the CHAMMI
- `--workers`: Number of parallel workers to use for downloading
- `--guidance`: If provided, the script will also download the guidance files along with the main data files.


We would like to thank the AWS Open Data Sponsorship Program for hosting our dataset. For more information, please visit: https://registry.opendata.aws/chammi/