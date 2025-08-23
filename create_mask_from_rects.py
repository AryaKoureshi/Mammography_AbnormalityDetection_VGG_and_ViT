import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import os
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import csv
from collections import defaultdict

# Function to save masks
def save_masks(image, masks, scores, image_name, output_dir):
    # Combine all masks into one mask
    combined_mask = np.any(masks, axis=0)
    
    # Convert the mask to uint8 format
    mask_uint8 = (combined_mask * 255).astype(np.uint8)
    
    # Save the mask
    mask_image = Image.fromarray(mask_uint8)
    mask_filename = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_mask.png")
    mask_image.save(mask_filename)
    print(f"Saved mask for {image_name} at {mask_filename}")

# Dictionary to store boxes for each image
image_boxes = defaultdict(list)

# Path to your CSV file
csv_file_path = r"C:\Users\Arya\Downloads\Telegram Desktop\1\labels_my-project-name_2024-09-10-08-09-41.csv"

# Read the CSV file and extract boxes
with open(csv_file_path, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_name = row['image_name']
        bbox_x = float(row['bbox_x'])
        bbox_y = float(row['bbox_y'])
        bbox_width = float(row['bbox_width'])
        bbox_height = float(row['bbox_height'])
        
        # Calculate the box coordinates
        x0 = bbox_x
        y0 = bbox_y
        x1 = bbox_x + bbox_width
        y1 = bbox_y + bbox_height
        
        # Append the box to the list for the corresponding image
        image_boxes[image_name].append([x0, y0, x1, y1])

# Select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    # Use bfloat16 for the entire script
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # Turn on tf32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS."
    )

# Clear any existing Hydra context
GlobalHydra.instance().clear()

# Paths to your model checkpoint and configuration file
checkpoint = r"C:\Users\Arya\sam2\checkpoints\sam2.1_hiera_large.pt"
model_cfg = r"C:\Users\Arya\sam2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"

# Extract config directory and name
config_dir = os.path.dirname(model_cfg)
config_name = os.path.basename(model_cfg)

# Initialize Hydra with the config directory
with initialize_config_dir(config_dir=config_dir, version_base=None):
    # Build the model using the config name
    model = build_sam2(config_name, checkpoint, device=device)

# Initialize the predictor
predictor = SAM2ImagePredictor(model)

# Directory where your images are stored
image_dir = r"C:\Users\Arya\Downloads\Telegram Desktop\1"

# Output directory to save masks
output_dir = r"C:\Users\Arya\Downloads\Telegram Desktop\1\masks"
os.makedirs(output_dir, exist_ok=True)

# Process each image
for image_name, boxes in image_boxes.items():
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Image {image_name} not found in {image_dir}. Skipping.")
        continue

    # Load the image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Perform prediction for each box
    all_masks = []
    all_scores = []
    with torch.inference_mode():
        predictor.set_image(image_np)
        for box in boxes:
            # Convert box to NumPy array and ensure it's in the correct format
            box_np = np.array(box)

            # Perform prediction using the box as prompt
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_np,
                multimask_output=False,  # Set to False to get a single mask
            )
            all_masks.append(masks)
            all_scores.append(scores)

    # Combine masks from all boxes
    if all_masks:
        # Concatenate all masks along the first axis
        combined_masks = np.concatenate(all_masks, axis=0)
        combined_scores = np.concatenate(all_scores, axis=0)

        # Save the masks
        save_masks(image_np, combined_masks, combined_scores, image_name, output_dir)
    else:
        print(f"No masks generated for {image_name}")
