import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime

# Constants
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp']
SEQUENCE_PREFIX = "seq"

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

def is_already_renamed(filename):
    # Check if already starts with seq_XXX_
    return re.match(r"^seq_\d{3}_", filename)

def check_similarity_align_image_stack(img1_path, img2_path):
    """
    Check if two images are similar using align_image_stack.
    Returns True if they can be aligned (have enough control points), False otherwise.
    """
    # Create a temporary directory for output files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Prefix for aligned output images - we don't actually need them,
        # but align_image_stack requires -o or -p to run optimization.
        # Directing the prefix to the temp directory ensures no clutter.
        temp_output_prefix = os.path.join(tmp_dir, "align_result")

        # align_image_stack command:
        # -o specifies the PTO file name.
        # -p specifies the prefix for the remapped TIFF images.
        # We put both in the temporary directory.
        command = [
            'align_image_stack',
            '-o', f"{temp_output_prefix}.pto",
            '-p', temp_output_prefix,
            img1_path,
            img2_path
        ]
        try:
            # Run with a timeout to avoid hangs
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)

            # If it exits with 0, it found enough control points to try optimization
            if result.returncode == 0:
                # Double check for "After control points pruning there are only 0 control points"
                if "After control points pruning there are only 0 control points" in result.stderr:
                    return False
                return True
            return False
        except subprocess.TimeoutExpired:
            print(f"Timeout checking similarity between {os.path.basename(img1_path)} and {os.path.basename(img2_path)}")
            return False
        except Exception as e:
            print(f"Error running align_image_stack: {e}")
            return False

def rename_images(directory):
    # Filter and sort images by modification time
    try:
        all_files = os.listdir(directory)
    except OSError as e:
        print(f"Error listing directory {directory}: {e}")
        return

    images = [f for f in all_files if is_image_file(f) and not is_already_renamed(f)]
    images.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))

    if len(images) < 2:
        print("Not enough images to process.")
        return

    print(f"Analyzing {len(images)} images in {directory} using align_image_stack...")

    sequences = []
    i = 0
    while i < len(images):
        current_img_name = images[i]
        current_img_path = os.path.join(directory, current_img_name)
        current_seq = [current_img_name]

        j = i + 1
        while j < len(images):
            next_img_name = images[j]
            next_img_path = os.path.join(directory, next_img_name)

            print(f"Checking similarity: {current_img_name} vs {next_img_name}...", end=" ", flush=True)
            is_similar = check_similarity_align_image_stack(current_img_path, next_img_path)

            if is_similar:
                print("SIMILAR")
                current_seq.append(next_img_name)
                # In sequences, we compare with the previous image to handle slow motion/drifts
                current_img_path = next_img_path
                current_img_name = next_img_name
                j += 1
            else:
                print("DIFFERENT")
                break

        if len(current_seq) > 1:
            sequences.append(current_seq)
            i = j
        else:
            i += 1

    if not sequences:
        print("No sequences of similar images found.")
        return

    print(f"\nFound {len(sequences)} sequences.")

    for seq_idx, seq in enumerate(sequences, 1):
        print(f"Processing sequence {seq_idx} ({len(seq)} images)...")
        for img_idx, img_name in enumerate(seq, 1):
            # Naming format: seq_{seq_id:03d}_{img_id:03d}_{original_name}
            new_name = f"seq_{seq_idx:03d}_{img_idx:03d}_{img_name}"

            old_path = os.path.join(directory, img_name)
            new_path = os.path.join(directory, new_name)

            if os.path.exists(new_path):
                print(f"Warning: {new_path} already exists. Skipping {img_name}.")
                continue

            try:
                mtime = os.path.getmtime(old_path)
                shutil.move(old_path, new_path)
                # Preserve modification time exactly as it was
                os.utime(new_path, (mtime, mtime))
                print(f"Renamed {img_name} -> {new_name}")
            except Exception as e:
                print(f"Error renaming {img_name}: {e}")

if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    rename_images(target_dir)
