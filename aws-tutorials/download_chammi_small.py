import json
import os
import argparse
import requests
import time
import random
from pprint import pprint
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import pandas as pd
from urllib.parse import quote


def parse_args():
    parser = argparse.ArgumentParser(description="Download CHAMMI-75 dataset images")
    parser.add_argument(
        "--download-folder",
        type=str,
        required=True,
        help="Folder path where images will be downloaded"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers for downloading (default: 16)"
    )
    parser.add_argument(
        "--guidance",
        action="store_true",
        help="If set, also download guidance files (.safetensors format)"
    )
    return parser.parse_args()


def download_file(url, local_path, timeout=30, allow_404=False, max_retries=5):
    """Download a single file from URL to local path with retry logic."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if allow_404 and response.status_code == 404:
                return True, f"{local_path}: skipped (not found)"
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return True, local_path
        except Exception as e:
            if allow_404 and "404" in str(e):
                return True, f"{local_path}: skipped (not found)"
            
            # Don't retry on last attempt
            if attempt == max_retries - 1:
                return False, f"{local_path}: {e}"
            
            # Exponential backoff with jitter
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)


def get_missing_images(clean_paths, download_folder):
    """Check which images are missing from the download folder."""
    missing = []
    for path in clean_paths:
        local_path = os.path.join(download_folder, path)
        if not os.path.exists(local_path):
            missing.append(path)
    return missing


def get_missing_guidance(clean_paths, download_folder):
    """Check which safetensor files are missing from the download folder."""
    missing = []
    for path in clean_paths:
        safetensor_path = os.path.splitext(path)[0] + ".safetensors"
        local_path = os.path.join(download_folder, safetensor_path)
        if not os.path.exists(local_path):
            missing.append(path)  # Return original path so download_guidance can convert it
    return missing


def download_image(path, base_url, download_folder):
    """Download a single image."""
    # Pre-encode the path for S3 - it expects %2B, %5B, %5D, etc.
    # quote() with safe='/' keeps forward slashes but encodes +, [, ]
    encoded_path = quote(path, safe='/')
    url = f"{base_url}/{encoded_path}"
    local_path = os.path.join(download_folder, path)
    return download_file(url, local_path)


def download_guidance(path, base_url, download_folder):
    """Download guidance file - convert .png path to .safetensor."""
    # Replace .png extension with .safetensor
    safetensor_path = os.path.splitext(path)[0] + ".safetensors"
    # Pre-encode the path for S3
    encoded_path = quote(safetensor_path, safe='/')
    url = f"{base_url}/{encoded_path}"
    local_path = os.path.join(download_folder, safetensor_path)
    return download_file(url, local_path, allow_404=True)


def main():
    args = parse_args()
    
    DOWNLOAD_FOLDER = args.download_folder
    NUM_WORKERS = args.workers
    INCLUDE_GUIDANCE = args.guidance
    
    # Define folder structure
    IMAGES_FOLDER = os.path.join(DOWNLOAD_FOLDER, "CHAMMI-75_small")
    GUIDANCE_FOLDER = os.path.join(DOWNLOAD_FOLDER, "CHAMMI-75_guidance")
    
    BASE_URL_TRAIN = "https://chammi-data.s3.amazonaws.com/CHAMMI-75/CHAMMI-75_train"
    BASE_URL_GUIDANCE = "https://chammi-data.s3.amazonaws.com/CHAMMI-75/CHAMMI-75_guidance"
    METADATA_URL = "https://chammi-data.s3.amazonaws.com/CHAMMI-75/CHAMMI-75_small_metadata.csv"
    
    # Create the download folders
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    if INCLUDE_GUIDANCE:
        os.makedirs(GUIDANCE_FOLDER, exist_ok=True)
    
    # Download metadata CSV to main folder
    print("Downloading metadata CSV...")
    metadata_local_path = os.path.join(DOWNLOAD_FOLDER, "CHAMMI-75_small_metadata.csv")
    success, result = download_file(METADATA_URL, metadata_local_path)
    if success:
        print(f"Metadata downloaded to: {metadata_local_path}")
    else:
        print(f"Failed to download metadata: {result}")
        return
    
    # Load the metadata
    df = pd.read_csv(metadata_local_path)
    
    # Get the clean paths
    clean_paths = df["storage.path"].str.split("/").apply(lambda x: "/".join(x[1:])).tolist()
    
    # Test first with a sample URL
    sample_path = clean_paths[0]
    sample_encoded_path = quote(sample_path, safe='/')
    sample_url = f"{BASE_URL_TRAIN}/{sample_encoded_path}"
    response = requests.get(sample_url)
    
    if response.status_code != 200:
        print(f"URL structure is wrong - check the path")
        print(f"Sample URL: {sample_url}")
        print(f"Status code: {response.status_code}")
        return
    
    print(f"Sample URL test passed: {response.status_code}")
    
    # Download images to CHAMMI-75_small folder
    # First check which images are missing
    print(f"\nChecking for missing images in {IMAGES_FOLDER}...")
    missing_images = get_missing_images(clean_paths, IMAGES_FOLDER)
    
    if not missing_images:
        print(f"All {len(clean_paths)} images already downloaded!")
    else:
        print(f"Found {len(missing_images)} missing images (out of {len(clean_paths)} total)")
        print(f"Downloading missing images with {NUM_WORKERS} workers...")
        failed_images = []
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(download_image, path, BASE_URL_TRAIN, IMAGES_FOLDER): path 
                for path in missing_images
            }
            
            for future in tqdm(as_completed(futures), total=len(missing_images), desc="Images"):
                success, result = future.result()
                if not success:
                    failed_images.append(result)
        
        if failed_images:
            print(f"\nFailed image downloads ({len(failed_images)}):")
            for f in failed_images[:10]:
                print(f"  {f}")
        else:
            print(f"\nAll {len(missing_images)} missing images downloaded successfully!")
    
    # Download guidance files if requested (in addition to images)
    if INCLUDE_GUIDANCE:
        # First check which safetensors are missing
        print(f"\nChecking for missing guidance files in {GUIDANCE_FOLDER}...")
        missing_guidance = get_missing_guidance(clean_paths, GUIDANCE_FOLDER)
        
        if not missing_guidance:
            print(f"All guidance files already downloaded!")
        else:
            print(f"Found {len(missing_guidance)} missing guidance files (out of {len(clean_paths)} total)")
            print(f"Downloading missing guidance files with {NUM_WORKERS} workers...")
            print("(Note: It's OK if some .safetensor files don't exist on server)")
            failed_guidance = []
            skipped_guidance = 0
            
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {
                    executor.submit(download_guidance, path, BASE_URL_GUIDANCE, GUIDANCE_FOLDER): path 
                    for path in missing_guidance
                }
                
                for future in tqdm(as_completed(futures), total=len(missing_guidance), desc="Guidance"):
                    success, result = future.result()
                    if not success:
                        failed_guidance.append(result)
                    elif "skipped" in str(result):
                        skipped_guidance += 1
            
            if failed_guidance:
                print(f"\nFailed guidance downloads ({len(failed_guidance)}):")
                for f in failed_guidance[:10]:
                    print(f"  {f}")
            else:
                downloaded_count = len(missing_guidance) - skipped_guidance
                print(f"\nGuidance download complete! {downloaded_count} files downloaded, {skipped_guidance} skipped (not found on server)")
    
    print("\nDownload complete!")


if __name__ == "__main__":
    main()