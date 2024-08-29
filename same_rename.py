import os
import re
import subprocess
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import tempfile

# Constants
OVERLAP_THRESHOLD = 0.9  # Define the threshold for high overlap
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']  # Supported image formats
SEQUENCE_PREFIX = "seq_"  # Prefix to identify already renamed files
ALIGN_PARAMS = [['--corr=0.8'], ['--corr=0.9'], ['--corr=0.7']]  # Different parameter sets for alignment

def is_image_file(filename):
    """Check if the file is an image based on its extension."""
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

def is_already_renamed(filename):
    """Check if the file is already renamed by the script."""
    return re.match(f"{SEQUENCE_PREFIX}[0-9]+_", filename)

def preserve_metadata(source, target):
    """Preserve the creation date and time of the source file to the target file."""
    creation_time = os.path.getmtime(source)
    os.utime(target, (creation_time, creation_time))

def run_align_image_stack(images, params, temp_pto_path):
    """Run align_image_stack with specified parameters and generate .pto file."""
    command = ['align_image_stack', '-o', temp_pto_path] + params + images
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        return parse_pto_file(temp_pto_path)
    except subprocess.CalledProcessError as e:
        print(f"Error running align_image_stack with params {params}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error running align_image_stack: {e}")
        return []

def parse_pto_file(pto_file):
    """Parse the .pto file to extract overlap scores based on control points."""
    overlap_scores = []
    try:
        with open(pto_file, 'r') as file:
            data = file.read()

        # Regex to find control points and other relevant overlap-related data in the .pto file
        control_point_pattern = re.compile(r'c n\d+ N\d+ x\d+ y\d+ X\d+ Y\d+ t(\d+)', re.MULTILINE)
        control_points = control_point_pattern.findall(data)

        # Calculate overlap score based on control point distribution and count
        control_point_count = len(control_points)
        if control_point_count > 0:
            overlap_density = control_point_count / max(1, len(set(control_points)))  # Normalize by unique point types
            overlap_scores.append(overlap_density)

    except FileNotFoundError:
        print(f"Error: .pto file {pto_file} not found.")
    except Exception as e:
        print(f"Error parsing .pto file {pto_file}: {e}")

    return overlap_scores

def get_overlap_ratios_parallel(images):
    """Calculate overlap ratios using parallel execution with different parameter sets."""
    overlap_results = []
    with ThreadPoolExecutor(max_workers=len(ALIGN_PARAMS)) as executor:
        # Generate unique temporary .pto file paths
        temp_files = [tempfile.NamedTemporaryFile(delete=False, suffix=".pto") for _ in ALIGN_PARAMS]
        temp_paths = [temp_file.name for temp_file in temp_files]
        futures = {
            executor.submit(run_align_image_stack, images, params, temp_path): (params, temp_path)
            for params, temp_path in zip(ALIGN_PARAMS, temp_paths)
        }
        for future in as_completed(futures):
            params, temp_path = futures[future]
            try:
                result = future.result()
                if result:
                    overlap_results.append(result)
            except Exception as e:
                print(f"Error with params {params}: {e}")
            finally:
                # Clean up temporary .pto files
                try:
                    os.remove(temp_path)
                except OSError as e:
                    print(f"Error deleting temporary file {temp_path}: {e}")

    # Flatten list of overlap scores and calculate statistics
    all_overlap_scores = [score for sublist in overlap_results for score in sublist]
    if all_overlap_scores:
        mean_overlap = np.mean(all_overlap_scores)
        median_overlap = np.median(all_overlap_scores)
        std_overlap = np.std(all_overlap_scores)
        print(f"Mean Overlap: {mean_overlap}, Median Overlap: {median_overlap}, Std Deviation: {std_overlap}")
        return mean_overlap, median_overlap, std_overlap
    else:
        print("No overlap scores found.")
        return None, None, None

def rename_images(directory):
    """Scan directory for images, find those with high overlap, and rename them sequentially."""
    images = [f for f in os.listdir(directory) if is_image_file(f) and not is_already_renamed(f)]
    images.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))  # Sort by creation time

    if len(images) < 2:
        print("Not enough images to process.")
        return

    # Calculate overlap ratios using parallel execution
    mean_overlap, median_overlap, std_overlap = get_overlap_ratios_parallel([os.path.join(directory, img) for img in images])

    if mean_overlap is None or median_overlap is None:
        print("Unable to compute valid overlap ratios.")
        return

    # Determine which images have high overlap based on statistical analysis
    high_overlap_images = [img for img in images if mean_overlap >= OVERLAP_THRESHOLD]

    # Rename images preserving order and metadata
    counter = 1
    for img in high_overlap_images:
        new_name = f"{SEQUENCE_PREFIX}{counter:03d}_{img}"
        new_path = os.path.join(directory, new_name)
        old_path = os.path.join(directory, img)
        if old_path != new_path:
            shutil.move(old_path, new_path)
            preserve_metadata(old_path, new_path)
            print(f"Renamed {img} to {new_name}")
        counter += 1

if __name__ == "__main__":
    directory = '.'  # Current directory
    rename_images(directory)
