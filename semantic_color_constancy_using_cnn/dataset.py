import typer
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import zipfile
import shutil
import os
import requests
from semantic_color_constancy_using_cnn.config import RAW_DATA_DIR

app = typer.Typer()

# Function to download the file using requests with a progress bar
def download_file(url: str, destination: Path):
    logger.info(f"Downloading file from {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB block size
    t = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()
    logger.success(f"Downloaded dataset to {destination}")

# Function to extract the ZIP file
def extract_zip(file_path: Path, extract_path: Path):
    logger.info(f"Extracting ZIP file: {file_path}")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        # Extract all files into a temporary directory first
        temp_extract_path = extract_path / "temp"
        zip_ref.extractall(temp_extract_path)

        # Move files from the temp folder to the final destination
        for item in temp_extract_path.iterdir():
            if item.is_dir():
                # Move contents of subdirectories directly to the target folder
                for sub_item in item.iterdir():
                    shutil.move(str(sub_item), str(extract_path))
                # Remove the now-empty directory
                item.rmdir()
            else:
                # Move individual files
                shutil.move(str(item), str(extract_path))

        # Clean up the temporary folder
        temp_extract_path.rmdir()
    logger.success(f"Extraction complete at {extract_path}")

@app.command()
def main(
    download_url: str = typer.Option("http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip", help="URL to download the ZIP folder"),
    output_folder: Path = typer.Option(RAW_DATA_DIR, help="Path for the output folder")
):
    # Create necessary directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download the ZIP file
    zip_file = RAW_DATA_DIR / "ADE20K.zip"
    download_file(download_url, zip_file)

    # Extract the ZIP folder
    extract_zip(zip_file, output_folder)

    # Clean up
    logger.info("Cleaning up temporary files")
    os.remove(zip_file)
    logger.success("Clean up complete")

if __name__ == "__main__":
    app()