import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from natsort import natsorted
import matplotlib.pyplot as plt  # For plotting
from torchinfo import summary
from torchview import draw_graph

import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define custom dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list, mask_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Ensure the image and mask have the same size
        if image.size != mask.size:
            print(f"Resizing image and mask from {image.size} and {mask.size} to same size.")
            # Resize mask to match image size
            mask = mask.resize(image.size, resample=Image.NEAREST)

        image = np.array(image)
        mask = np.array(mask)

        # Normalize mask to 0 and 1
        mask = mask / 255.0
        mask = mask.astype('float32')

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Convert to tensors
            image = transforms.ToTensor()(image)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask

# Define the UNet model (using a pre-trained backbone for better performance)
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
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Paths to your images and masks
    image_dir = r"C:\Users\Arya\Downloads\Telegram Desktop\1"
    mask_dir = r"C:\Users\Arya\Downloads\Telegram Desktop\1\masks"

    # Get list of image and mask files
    image_files = natsorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    mask_files = natsorted([f for f in os.listdir(mask_dir) if f.endswith('_mask.png')])

    # Ensure that the number of images and masks match
    assert len(image_files) == len(mask_files), "Number of images and masks do not match!"

    # Split data into training and validation sets
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_files, mask_files, test_size=0.3, random_state=42
    )

    print(f'Total images: {len(image_files)}')
    print(f'Training images: {len(train_images)}')
    print(f'Validation images: {len(val_images)}')

    # Data augmentation transformations for training data
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5),
        A.GaussianBlur(p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Transformations for validation data (only normalization)
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Create datasets
    train_dataset = SegmentationDataset(image_dir, mask_dir, train_images, train_masks, transform=train_transform)
    val_dataset = SegmentationDataset(image_dir, mask_dir, val_images, val_masks, transform=val_transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Instantiate the model, loss function, and optimizer
    model = UNet(n_classes=1).to(device)

    # Use BCEWithLogitsLoss for binary segmentation
    criterion = nn.BCEWithLogitsLoss()

    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Learning rate scheduler (optional)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Lists to store losses and accuracies
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    num_epochs = 40
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            # Calculate accuracy
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == masks.byte()).sum().item()
            train_total += masks.numel()

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                outputs = outputs.squeeze(1)

                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                # Calculate accuracy
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == masks.byte()).sum().item()
                val_total += masks.numel()

        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Training Loss: {train_loss:.4f}, '
              f'Training Acc: {train_accuracy:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Validation Acc: {val_accuracy:.4f}')

        # Save the model checkpoint if validation loss decreases
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), image_dir + "\\best_model_unet.pth")
            print('Model saved!')

    print('Training complete.')

    # Plotting the losses and accuracies
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    # Viz model
    print(summary(model, input_size=(1, 3, 256, 256)))
    model_graph = draw_graph(model, input_size=(1, 3, 256, 256))
    model_graph.visual_graph.attr(dpi='600')  # You can set a higher DPI if needed
    model_graph.visual_graph.format = 'png'  # Set the output format to PNG
    model_graph.visual_graph.render('model_graph', view=False, cleanup=True)
    
if __name__ == '__main__':
    main()



