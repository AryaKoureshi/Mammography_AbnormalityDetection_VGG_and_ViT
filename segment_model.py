import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Define the UNet model (using the same architecture as your trained model)
class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1)
        self.conv_up2 = nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1)
        self.conv_up1 = nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1)
        self.conv_up0 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)

        self.conv_original_size0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv_original_size1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_original_size2 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)

        self.conv_last = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.upsample(layer4)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

def main():
    # Specify the paths
    model_path = r"C:\Users\Arya\Downloads\Telegram Desktop\1\best_model_unet.pth"  # Update with your model path
    image_path = r"C:\Users\Arya\Downloads\Telegram Desktop\1\49.png"  # Update with the path to your image
    output_path = r"C:\Users\Arya\Downloads\Telegram Desktop\1\segmented_images\49_segmented.png"  # Update with the desired output path

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Instantiate the model
    model = UNet(n_classes=1).to(device)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Define the transformations (same as validation transforms)
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Load and preprocess the image
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size  # Keep the original size for resizing later
    image_np = np.array(original_image)
    transformed = val_transform(image=image_np)
    input_tensor = transformed['image'].unsqueeze(0).to(device)  # Add batch dimension

    # Pass the image through the model
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output).squeeze().cpu().numpy()

    # Post-process the output mask
    mask = (output > 0.5).astype(np.uint8)  # Threshold the output
    mask_resized = Image.fromarray(mask * 255).resize(original_size, resample=Image.NEAREST)
    mask_resized_np = np.array(mask_resized) // 255  # Ensure binary mask

    # Overlay the mask onto the original image
    overlayed_image = overlay_mask_on_image(original_image, mask_resized_np, color=(255, 0, 0), alpha=0.5)

    # Save the resulting image
    icc_profile = original_image.info.get('icc_profile')  # Preserve ICC profile if present
    overlayed_image.save(output_path, format='PNG', compress_level=0, icc_profile=icc_profile)
    print(f"Overlayed image saved to {output_path}")

    # Optionally display the image
    # overlayed_image.show()

def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    Overlay the mask on the image with the given color and alpha transparency.
    :param image: PIL Image
    :param mask: NumPy array (binary mask)
    :param color: Tuple of RGB values (default is red)
    :param alpha: Transparency level (0.0 to 1.0)
    :return: PIL Image with mask overlay
    """
    # Convert mask to PIL Image
    mask = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')

    # Create a color image for the mask
    color_mask = Image.new('RGB', image.size, color)
    # Apply the mask to get the colored mask
    colored_mask = Image.composite(color_mask, Image.new('RGB', image.size), mask)

    # Blend the original image and the colored mask
    blended = Image.blend(image, colored_mask, alpha)

    return blended

if __name__ == '__main__':
    main()
