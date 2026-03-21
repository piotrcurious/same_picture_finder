import os
import re
import shutil
import cv2
import numpy as np
from datetime import datetime

# Constants
SIMILARITY_THRESHOLD = 0.9  # Adjusted threshold
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp']
SEQUENCE_PREFIX = "seq"

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

def is_already_renamed(filename):
    # Check if already starts with seq_XXX_
    return re.match(r"^seq_\d{3}_", filename)

def get_image_fingerprint(image_path):
    """Generate a fingerprint of the image for similarity comparison."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        # Resize to a small fixed size and convert to grayscale
        img = cv2.resize(img, (64, 64)) # Slightly larger for better detail
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply slight Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def calculate_similarity(img1, img2):
    """Calculate structural similarity or correlation between two fingerprints."""
    if img1 is None or img2 is None:
        return 0
    # Use correlation coefficient
    res = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    return res[0][0]

def rename_images(directory):
    # Filter and sort images by modification time
    all_files = os.listdir(directory)
    images = [f for f in all_files if is_image_file(f) and not is_already_renamed(f)]
    images.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))

    if len(images) < 2:
        print("Not enough images to process.")
        return

    print(f"Analyzing {len(images)} images in {directory}...")

    fingerprints = {}
    for img in images:
        fp = get_image_fingerprint(os.path.join(directory, img))
        if fp is not None:
            fingerprints[img] = fp

    valid_images = [img for img in images if img in fingerprints]

    if not valid_images:
        print("No valid images found.")
        return

    sequences = []
    i = 0
    while i < len(valid_images):
        current_img = valid_images[i]
        current_seq = [current_img]

        j = i + 1
        while j < len(valid_images):
            next_img = valid_images[j]
            similarity = calculate_similarity(fingerprints[current_img], fingerprints[next_img])

            # Use a slightly lower threshold for consecutive images in a sequence
            # to handle gradual changes, but keep it high enough to avoid merging different scenes.
            if similarity >= SIMILARITY_THRESHOLD:
                current_seq.append(next_img)
                current_img = next_img # Compare with the last added image
                j += 1
            else:
                break

        if len(current_seq) > 1:
            sequences.append(current_seq)
            i = j
        else:
            i += 1

    if not sequences:
        print("No sequences of similar images found.")
        return

    print(f"Found {len(sequences)} sequences.")

    for seq_idx, seq in enumerate(sequences, 1):
        print(f"Processing sequence {seq_idx} ({len(seq)} images)...")
        for img_idx, img_name in enumerate(seq, 1):
            ext = os.path.splitext(img_name)[1]
            # Naming format: seq_XXX_YYY_original_name.ext
            new_name = f"seq_{seq_idx:03d}_{img_idx:03d}_{img_name}"

            old_path = os.path.join(directory, img_name)
            new_path = os.path.join(directory, new_name)

            if os.path.exists(new_path):
                print(f"Warning: {new_path} already exists. Skipping {img_name}.")
                continue

            try:
                mtime = os.path.getmtime(old_path)
                shutil.move(old_path, new_path)
                os.utime(new_path, (mtime, mtime))
                print(f"Renamed {img_name} -> {new_name}")
            except Exception as e:
                print(f"Error renaming {img_name}: {e}")

if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    rename_images(target_dir)
