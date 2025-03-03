from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)]
        self.transform = transform
        self.label = [1.0]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.label)
        return image, label


def custom_transform(channel_size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1)
    ])


def create_custom_data_loader(channel_size, dir_path, batch_size, num_workers=4):
    transform = custom_transform(channel_size)
    custom_dataset = CustomDataset(dir_path, transform=transform)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return data_loader, custom_dataset


def show_tensor_image(image, cmap=None, label=None):
    permuted = image.permute(1, 2, 0)

    plt.imshow(permuted, cmap=cmap)

    if label is not None:
        plt.title(label)

    plt.axis('off')

    plt.show()


import math


def show_tensor_images(images, labels=None, cmap=None, save=None, filename=None, to_cpu=True, show=True):
    n_images = len(images)
    n_rows = math.ceil(math.sqrt(n_images))
    n_cols = math.ceil(n_images / n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()

    for i, (image, ax) in enumerate(zip(images, axes)):
        if to_cpu:
            image = image.cpu()
        permuted = image.permute(1, 2, 0).clamp(0, 1)

        ax.imshow(permuted, cmap=cmap)
        if labels is not None:
            ax.set_title(labels[i])
        ax.axis('off')

    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    if save:
        os.makedirs(save, exist_ok=True)
        save_path = os.path.join(save, filename)
        plt.savefig(save_path)

    if show:
        plt.show()


